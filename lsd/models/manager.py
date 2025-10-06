import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .components import HiddenStatesExtractor, TruthEncoder, build_enhanced_projection_heads, enhanced_contrastive_loss
from ..data.manager import TextPairDataset
from ..core.config import LayerwiseSemanticDynamicsConfig
from ..utils.helpers import DEVICE, logger, dir_manager

class ModelManager:
    """Enhanced model manager with better training strategies."""
    
    def __init__(self, config: LayerwiseSemanticDynamicsConfig):
        self.config = config
        self.extractor = None
        self.truth_encoder = None
        self.hidden_proj = None
        self.truth_proj = None
        
    def initialize_models(self):
        """Initialize all models with proper dimensions and architecture"""
        logger.log("Initializing enhanced models...")
        
        # Initialize hidden states extractor
        self.extractor = HiddenStatesExtractor(
            model_name=self.config.model_name,
            max_length=self.config.max_length
        )
        
        # Initialize truth encoder
        self.truth_encoder = TruthEncoder(name=self.config.truth_encoder_name)
        
        # Get dimensions from models
        with torch.no_grad():
            dummy_text = ["test sentence for dimension checking"]
            dummy_hidden = self.extractor.get_hidden_states(dummy_text)
            hidden_dim = dummy_hidden.shape[-1]
            
            dummy_truth = self.truth_encoder.encode_batch(dummy_text)
            truth_dim = dummy_truth.shape[-1]
        
        logger.log(f"Model dimensions - Hidden: {hidden_dim}, Truth: {truth_dim}")
        
        # Build enhanced projection heads
        self.hidden_proj, self.truth_proj = build_enhanced_projection_heads(
            hidden_dim, truth_dim, self.config.shared_dim
        )
        
        # Initialize weights properly
        self._initialize_weights()
        
        return self.extractor, self.truth_encoder, self.hidden_proj, self.truth_proj
    
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for module in [self.hidden_proj, self.truth_proj]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def load_pretrained(self, path: str = None):
        """Load pretrained models with validation."""
        path = path or self.config.pretrained_path
        model_dir = Path(path)
        
        if not model_dir.exists():
            logger.log(f"No pretrained models found at {path}", "WARNING")
            return False
        
        try:
            # Initialize models first
            self.initialize_models()
            
            # Load state dicts
            hidden_proj_path = dir_manager.get_model_path("hidden_proj_best")
            truth_proj_path = dir_manager.get_model_path("truth_proj_best")
            
            if hidden_proj_path.exists() and truth_proj_path.exists():
                self.hidden_proj.load_state_dict(torch.load(hidden_proj_path, map_location=DEVICE))
                self.truth_proj.load_state_dict(torch.load(truth_proj_path, map_location=DEVICE))
                logger.log("Successfully loaded pretrained models")
                return True
            else:
                logger.log("Pretrained model files not found", "WARNING")
                return False
                
        except Exception as e:
            logger.log(f"Error loading pretrained models: {e}", "ERROR")
            # Reinitialize if loading fails
            self.initialize_models()
            return False
    
    def save_models(self, suffix: str = "best"):
        """Save current models with given suffix"""
        torch.save(self.hidden_proj.state_dict(), dir_manager.get_model_path(f"hidden_proj_{suffix}"))
        torch.save(self.truth_proj.state_dict(), dir_manager.get_model_path(f"truth_proj_{suffix}"))
        logger.log(f"Models saved with suffix: {suffix}")

def train_projection_heads(config: LayerwiseSemanticDynamicsConfig, 
                          pairs: List[Tuple[str, str, str]]) -> ModelManager:
    """Enhanced training with better strategies and comprehensive logging."""
    model_manager = ModelManager(config)
    
    # Check if we should use pretrained models
    if config.use_pretrained and model_manager.load_pretrained():
        logger.log("Using pretrained models, skipping training")
        return model_manager
    
    # Initialize models
    extractor, truth_encoder, hidden_proj, truth_proj = model_manager.initialize_models()
    
    # Split data into train and validation sets
    n = int(config.train_test_split * len(pairs))
    train_pairs, val_pairs = pairs[:n], pairs[n:]
    
    # Verify class distribution
    train_factual = sum(1 for p in train_pairs if p[2] == 'factual')
    train_hallucination = len(train_pairs) - train_factual
    logger.log(f"Training distribution - Factual: {train_factual}, Hallucination: {train_hallucination}")
    
    # Create datasets and dataloaders
    train_dataset = TextPairDataset(train_pairs)
    val_dataset = TextPairDataset(val_pairs)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, collate_fn=lambda x: x)
    
    logger.log(f"Training on {len(train_pairs)} samples, validating on {len(val_pairs)} samples")
    
    # Setup optimizer with AdamW
    optimizer = torch.optim.AdamW(
        list(hidden_proj.parameters()) + list(truth_proj.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    
    # Training history tracking
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 'val_accuracy': [],
        'train_pos_sim': [], 'train_neg_sim': [], 'learning_rate': []
    }
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    logger.log("Starting enhanced training...")
    
    # Main training loop
    for epoch in range(1, config.epochs + 1):
        # Training phase
        hidden_proj.train()
        truth_proj.train()
        train_losses = []
        pos_similarities = []
        neg_similarities = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}"):
            texts = [item["text"] for item in batch]
            truths = [item["truth"] for item in batch]
            labels = torch.tensor([item["label"] for item in batch]).to(DEVICE)
            
            # Get embeddings (no gradient for base models)
            with torch.no_grad():
                hidden_states = extractor.get_hidden_states(texts)
                last_layer_hidden = hidden_states[:, -1, :]  # Use last layer
                truth_embeddings = truth_encoder.encode_batch(truths)
            
            # Project to shared space and normalize
            hidden_projected = F.normalize(hidden_proj(last_layer_hidden), p=2, dim=-1)
            truth_projected = F.normalize(truth_proj(truth_embeddings), p=2, dim=-1)
            
            # Compute cosine similarities
            cos_sim = F.cosine_similarity(hidden_projected, truth_projected, dim=-1)
            
            # Track similarities for monitoring
            if labels.sum() > 0:
                pos_similarities.extend(cos_sim[labels == 1].cpu().tolist())
            if (1 - labels).sum() > 0:
                neg_similarities.extend(cos_sim[labels == 0].cpu().tolist())
            
            # Compute contrastive loss
            loss = enhanced_contrastive_loss(cos_sim, labels, config.margin)
            train_losses.append(loss.item())
            
            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                list(hidden_proj.parameters()) + list(truth_proj.parameters()),
                max_norm=1.0
            )
            
            optimizer.step()
        
        # Validation phase
        hidden_proj.eval()
        truth_proj.eval()
        val_losses = []
        val_accuracies = []
        
        with torch.no_grad():
            for batch in val_loader:
                texts = [item["text"] for item in batch]
                truths = [item["truth"] for item in batch]
                labels = torch.tensor([item["label"] for item in batch]).to(DEVICE)
                
                hidden_states = extractor.get_hidden_states(texts)
                last_layer_hidden = hidden_states[:, -1, :]
                truth_embeddings = truth_encoder.encode_batch(truths)
                
                hidden_projected = F.normalize(hidden_proj(last_layer_hidden), p=2, dim=-1)
                truth_projected = F.normalize(truth_proj(truth_embeddings), p=2, dim=-1)
                
                cos_sim = F.cosine_similarity(hidden_projected, truth_projected, dim=-1)
                loss = enhanced_contrastive_loss(cos_sim, labels, config.margin)
                val_losses.append(loss.item())
                
                # Calculate accuracy
                preds = (cos_sim > 0).float()
                accuracy = (preds == labels).float().mean().item()
                val_accuracies.append(accuracy)
        
        # Compute epoch statistics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_accuracy = np.mean(val_accuracies)
        current_lr = scheduler.get_last_lr()[0]
        
        # Similarity analysis
        train_pos_sim = np.mean(pos_similarities) if pos_similarities else 0
        train_neg_sim = np.mean(neg_similarities) if neg_similarities else 0
        
        scheduler.step()
        
        # Update history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['train_pos_sim'].append(train_pos_sim)
        history['train_neg_sim'].append(train_neg_sim)
        history['learning_rate'].append(current_lr)
        
        # Enhanced logging
        logger.log(f"Epoch {epoch}: "
                  f"train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, "
                  f"val_acc={val_accuracy:.3f}, "
                  f"pos_sim={train_pos_sim:.3f}, "
                  f"neg_sim={train_neg_sim:.3f}, "
                  f"lr={current_lr:.2e}")
        
        # Early stopping with model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_manager.save_models("best")
            patience_counter = 0
            logger.log(f"New best model saved with val_loss={val_loss:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.log(f"Early stopping at epoch {epoch}")
            break
    
    # Save final models and training history
    model_manager.save_models("final")
    pd.DataFrame(history).to_csv(dir_manager.results_dir / "enhanced_training_history.csv", index=False)
    
    logger.log("Enhanced training completed successfully")
    return model_manager