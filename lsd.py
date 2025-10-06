# ==============================================================
# COMPLETE FIXED LAYER-WISE SEMANTIC DYNAMICS IMPLEMENTATION
# All JSON serialization issues and type conversion bugs fixed
# ==============================================================

import os, math, random
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Union
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from scipy.stats import ttest_ind
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import datasets

# Enhanced metrics and evaluation
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, accuracy_score, 
    classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc,
    matthews_corrcoef, cohen_kappa_score, average_precision_score
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# ==============================================================
# JSON Serialization Helper Functions
# ==============================================================

def convert_to_json_serializable(obj):
    """
    Convert numpy/pandas types to JSON-serializable Python types.
    Handles nested dictionaries, lists, and arrays recursively.
    """
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    elif isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def safe_json_dump(obj, filepath):
    """
    Safely dump object to JSON file with type conversion.
    """
    serializable_obj = convert_to_json_serializable(obj)
    with open(filepath, 'w') as f:
        json.dump(serializable_obj, f, indent=2)

# ==============================================================
# Enhanced Configuration System
# ==============================================================

class OperationMode(Enum):
    """Operation modes for the analysis system"""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    HYBRID = "hybrid"

class DatasetSource(Enum):
    """Available dataset sources"""
    SYNTHETIC = "synthetic"
    TRUTHFULQA = "truthfulqa"
    FEVER = "fever"
    CUSTOM = "custom"

@dataclass
class LayerwiseSemanticDynamicsConfig:
    """
    Enhanced configuration for Layer-wise Semantic Dynamics analysis.
    Controls all aspects of training, evaluation, and analysis.
    """
    
    # Core model settings
    model_name: str = "gpt2"
    truth_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    shared_dim: int = 256
    max_length: int = 128
    
    # Training parameters
    batch_size: int = 8
    epochs: int = 30
    learning_rate: float = 5e-5
    margin: float = 0.5
    weight_decay: float = 1e-5
    
    # Data settings
    num_pairs: int = 1000
    datasets: List[str] = None
    train_test_split: float = 0.8
    cross_validation_folds: int = 5
    
    # Operation mode
    mode: OperationMode = OperationMode.HYBRID
    use_pretrained: bool = False
    pretrained_path: str = "lsd_trained_enhanced"
    
    # Evaluation settings
    metrics: List[str] = None
    confidence_threshold: float = 0.7
    composite_score_weights: Dict[str, float] = None
    
    # Advanced settings
    enable_clustering: bool = True
    enable_anomaly_detection: bool = True
    enable_confidence_calibration: bool = True
    enable_ensemble: bool = True
    
    def __post_init__(self):
        """Initialize default values for None fields"""
        if self.datasets is None:
            self.datasets = ["synthetic", "truthfulqa"]
        
        if self.metrics is None:
            self.metrics = [
                'f1', 'auroc', 'precision', 'recall', 'specificity', 
                'accuracy', 'mcc', 'kappa', 'prauc', 'f2'
            ]
        
        if self.composite_score_weights is None:
            self.composite_score_weights = {
                'f1': 0.25,
                'auroc': 0.20,
                'precision': 0.15,
                'recall': 0.15,
                'specificity': 0.10,
                'mcc': 0.15
            }

# ==============================================================
# Global Setup and Utilities
# ==============================================================

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DirectoryManager:
    """
    Manages directory structure for the analysis system.
    Creates and maintains organized folders for models, plots, results, data, and cache.
    """
    def __init__(self, base_dir: str = "layerwise_semantic_dynamics_system"):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.plots_dir = self.base_dir / "plots"
        self.results_dir = self.base_dir / "results"
        self.data_dir = self.base_dir / "data"
        self.cache_dir = self.base_dir / "cache"
        
        # Create all directories
        for directory in [self.models_dir, self.plots_dir, self.results_dir, 
                         self.data_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name: str) -> Path:
        """Get path for model file"""
        return self.models_dir / f"{model_name}.pt"
    
    def get_plot_path(self, plot_name: str) -> Path:
        """Get path for plot file"""
        return self.plots_dir / f"{plot_name}.png"
    
    def get_result_path(self, result_name: str) -> Path:
        """Get path for result file"""
        return self.results_dir / f"{result_name}.json"

# Initialize global directory manager
dir_manager = DirectoryManager()

class EnhancedLogger:
    """
    Enhanced logging system with different levels and file persistence.
    Supports DEBUG, INFO, WARNING, and ERROR levels.
    """
    def __init__(self, log_file: Path = None, level: str = "INFO"):
        self.log_file = log_file or dir_manager.base_dir / "execution.log"
        self.level = level
        self.levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        
    def log(self, message: str, level: str = "INFO", print_message: bool = True):
        """Log message with timestamp and level"""
        if self.levels[level] >= self.levels[self.level]:
            timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] [{level}] {message}"
            
            with open(self.log_file, "a") as f:
                f.write(log_entry + "\n")
            
            if print_message:
                print(log_entry)

# Initialize global logger
logger = EnhancedLogger()

# ==============================================================
# Core Model Components
# ==============================================================

def mean_pool_hidden(hidden, attn_mask):
    """
    Mean pool hidden states using attention mask.
    Properly handles variable-length sequences.
    """
    mask = attn_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

class HiddenStatesExtractor:
    """
    Extracts hidden states from all layers of a language model.
    Supports caching for performance and multiple layer extraction strategies.
    """
    def __init__(self, model_name: str, device: str = DEVICE, max_length: int = 128,
                 use_cache: bool = True, layer_strategy: str = "all"):
        self.device = device
        self.use_cache = use_cache
        self.layer_strategy = layer_strategy
        
        logger.log(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.log(f"Loading model {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            output_hidden_states=True,
            torch_dtype=torch.float32
        ).to(device).eval()
        
        self.max_length = max_length
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        
        # Cache for performance
        self._cache = {}
        
        logger.log(f"Initialized HiddenStatesExtractor with {self.num_layers} layers")

    def get_hidden_states(self, texts: List[str]) -> torch.Tensor:
        """
        Extract hidden states from all layers for given texts.
        Returns tensor of shape [batch_size, num_layers, hidden_size]
        """
        cache_key = hash(tuple(texts)) if self.use_cache else None
        
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Tokenize inputs
        toks = self.tokenizer(
            texts, return_tensors="pt", truncation=True, 
            padding=True, max_length=self.max_length
        ).to(self.device)
        
        # Get hidden states from all layers
        with torch.no_grad():
            outs = self.model(**toks)
        
        # Stack all hidden states and mean pool
        hidden_states = []
        for layer_hidden in outs.hidden_states:
            pooled = mean_pool_hidden(layer_hidden, toks["attention_mask"])
            hidden_states.append(pooled)
        
        result = torch.stack(hidden_states, dim=1)  # [batch, layers, hidden_dim]
        
        if self.use_cache and cache_key:
            self._cache[cache_key] = result
            
        return result

    def clear_cache(self):
        """Clear the hidden states cache"""
        self._cache.clear()

class TruthEncoder:
    """
    Encodes truth statements using sentence transformers.
    Provides normalized embeddings for semantic similarity comparison.
    """
    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 device: str = DEVICE, use_cache: bool = True):
        logger.log(f"Loading truth encoder: {name}")
        self.model = SentenceTransformer(name).to(device)
        self.use_cache = use_cache
        self._cache = {}
        logger.log(f"Initialized TruthEncoder: {name}")
        
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts with caching and L2 normalization.
        Returns normalized embeddings.
        """
        cache_key = hash(tuple(texts)) if self.use_cache else None
        
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        emb = self.model.encode(texts, convert_to_tensor=True).to(DEVICE)
        result = F.normalize(emb, p=2, dim=-1)
        
        if self.use_cache and cache_key:
            self._cache[cache_key] = result
            
        return result
    
    def clear_cache(self):
        """Clear the encoding cache"""
        self._cache.clear()

def build_enhanced_projection_heads(hidden_dim: int, truth_dim: int, 
                                  shared_dim: int) -> Tuple[nn.Module, nn.Module]:
    """
    Build enhanced projection heads with deep architecture.
    Uses GELU activation, dropout, and layer normalization for better training.
    
    Args:
        hidden_dim: Dimension of hidden states from language model
        truth_dim: Dimension of truth encoder embeddings
        shared_dim: Target shared embedding dimension
    
    Returns:
        Tuple of (hidden_proj, truth_proj) neural networks
    """
    
    # Intermediate dimensions for deeper network
    hidden_dims = [shared_dim * 4, shared_dim * 2]
    
    # Hidden states projection network
    hidden_proj = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dims[0]),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.LayerNorm(hidden_dims[0]),
        
        nn.Linear(hidden_dims[0], hidden_dims[1]),
        nn.GELU(), 
        nn.Dropout(0.1),
        nn.LayerNorm(hidden_dims[1]),
        
        nn.Linear(hidden_dims[1], shared_dim),
        nn.LayerNorm(shared_dim)
    ).to(DEVICE)
    
    # Truth embeddings projection network
    truth_proj = nn.Sequential(
        nn.Linear(truth_dim, hidden_dims[0]),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.LayerNorm(hidden_dims[0]),
        
        nn.Linear(hidden_dims[0], hidden_dims[1]),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.LayerNorm(hidden_dims[1]),
        
        nn.Linear(hidden_dims[1], shared_dim),
        nn.LayerNorm(shared_dim)
    ).to(DEVICE)
    
    logger.log(f"Built enhanced projection heads with dimensions: {hidden_dims}")
    return hidden_proj, truth_proj

# ==============================================================
# Enhanced Data Management
# ==============================================================

class DataManager:
    """
    Comprehensive data manager supporting multiple dataset sources.
    Handles synthetic data generation, TruthfulQA loading, and class balancing.
    """
    def __init__(self, config: LayerwiseSemanticDynamicsConfig):
        self.config = config
        
    def _load_synthetic_pairs(self) -> List[Tuple[str, str, str]]:
        """
        Generate comprehensive synthetic dataset with factual and hallucination pairs.
        Returns list of (text, truth, label) tuples.
        """
        pairs = [
            # Factual pairs - semantically aligned
            ("The Earth orbits the Sun.", "Earth revolves around the Sun once a year.", "factual"),
            ("Water boils at 100°C at sea level.", "Water boils at 100 degrees Celsius.", "factual"),
            ("Photosynthesis produces oxygen.", "Plants release oxygen during photosynthesis.", "factual"),
            ("The human body has 206 bones.", "An adult human skeleton typically consists of 206 bones.", "factual"),
            ("Shakespeare wrote Hamlet.", "William Shakespeare is the author of the play Hamlet.", "factual"),
            
            # Hallucination pairs - semantically misaligned
            ("The Earth orbits the Moon.", "Earth revolves around the Sun yearly.", "hallucination"),
            ("Water boils at 50°C at sea level.", "Water requires 100°C to boil.", "hallucination"),
            ("Photosynthesis consumes oxygen.", "Plants produce oxygen.", "hallucination"),
            ("The human body has 300 bones.", "An adult human has 206 bones.", "hallucination"),
            ("Shakespeare wrote Harry Potter.", "J.K. Rowling wrote Harry Potter.", "hallucination"),
        ]
        
        # Scale the dataset to desired size
        scaled_pairs = []
        for i in range(self.config.num_pairs // 10):
            for pair in pairs:
                scaled_pairs.append(pair)
                
        return scaled_pairs[:self.config.num_pairs]
    
    def load_truthfulqa(self) -> List[Tuple[str, str, str]]:
        """
        Load TruthfulQA dataset from HuggingFace.
        Extracts question-answer pairs with correct and incorrect answers.
        """
        pairs = []
        try:
            logger.log("Loading TruthfulQA dataset...")
            dataset = datasets.load_dataset("truthful_qa", "generation", split="validation")
            
            for example in dataset:
                question = example.get("question", "").strip()
                correct_answers = example.get("best_answer", "")
                incorrect_answers = example.get("incorrect_answers", [])
                
                # Add factual pairs from correct answers
                if question and correct_answers:
                    pairs.append((question, correct_answers, "factual"))
                
                # Add hallucination pairs from incorrect answers
                for incorrect_answer in incorrect_answers[:1]:  # Use first incorrect answer
                    if question and incorrect_answer:
                        pairs.append((question, incorrect_answer, "hallucination"))
                        
        except Exception as e:
            logger.log(f"TruthfulQA load failed: {e}, using synthetic data only", "WARNING")
            
        return pairs
    
    def build_dataset(self) -> List[Tuple[str, str, str]]:
        """
        Build comprehensive dataset from multiple sources with class balancing.
        Returns balanced list of (text, truth, label) tuples.
        """
        all_pairs = []
        dataset_stats = {}
        
        # Load synthetic data if requested
        if "synthetic" in self.config.datasets:
            synthetic_pairs = self._load_synthetic_pairs()
            all_pairs.extend(synthetic_pairs)
            dataset_stats["synthetic"] = len(synthetic_pairs)
        
        # Load TruthfulQA data if requested
        if "truthfulqa" in self.config.datasets:
            truthfulqa_pairs = self.load_truthfulqa()
            all_pairs.extend(truthfulqa_pairs)
            dataset_stats["truthfulqa"] = len(truthfulqa_pairs)
        
        # Balance classes for fair training
        factuals = [p for p in all_pairs if p[2] == "factual"]
        hallucinations = [p for p in all_pairs if p[2] == "hallucination"]
        
        logger.log(f"Before balancing: {len(factuals)} factual, {len(hallucinations)} hallucination")
        
        # Take equal number from each class
        min_count = min(len(factuals), len(hallucinations))
        if min_count > 0:
            factuals = factuals[:min_count]
            hallucinations = hallucinations[:min_count]
            balanced_pairs = factuals + hallucinations
            random.shuffle(balanced_pairs)
            final_pairs = balanced_pairs[:self.config.num_pairs]
        else:
            final_pairs = all_pairs[:self.config.num_pairs]
        
        logger.log(f"Final dataset: {len(final_pairs)} pairs")
        logger.log(f"Dataset statistics: {dataset_stats}")
        
        return final_pairs

class TextPairDataset(Dataset):
    """
    PyTorch Dataset for text-truth pairs.
    Provides efficient loading and batching during training.
    """
    def __init__(self, pairs: List[Tuple[str, str, str]]):
        self.pairs = pairs
        
    def __len__(self) -> int:
        return len(self.pairs)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text, truth, label_str = self.pairs[idx]
        
        return {
            "text": text, 
            "truth": truth, 
            "label": 1 if label_str == "factual" else 0,
            "label_str": label_str,
            "idx": idx
        }

# ==============================================================
# Enhanced Training System
# ==============================================================

def enhanced_contrastive_loss(cos_sim: torch.Tensor, labels: torch.Tensor, 
                            margin: float = 0.5) -> torch.Tensor:
    """
    Enhanced contrastive loss with better gradient flow.
    
    For positive pairs (factual): maximize similarity (target = 1)
    For negative pairs (hallucination): minimize similarity (target <= -margin)
    
    Args:
        cos_sim: Cosine similarity scores [-1, 1]
        labels: Binary labels (1 = factual, 0 = hallucination)
        margin: Margin for negative pairs
    
    Returns:
        Scalar loss value
    """
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    # Positive pairs - we want high similarity (close to 1)
    pos_loss = (1 - cos_sim[pos_mask]).pow(2).mean() if pos_mask.any() else 0
    
    # Negative pairs - we want low similarity (<= -margin)
    neg_loss = F.relu(cos_sim[neg_mask] + margin).pow(2).mean() if neg_mask.any() else 0
    
    # Combine losses with equal weighting
    total_loss = 0.5 * pos_loss + 0.5 * neg_loss
    
    return total_loss

class ModelManager:
    """
    Enhanced model manager with better training strategies.
    Handles model initialization, loading, saving, and training orchestration.
    """
    
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
        """
        Load pretrained models with validation.
        Returns True if successful, False otherwise.
        """
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
    """
    Enhanced training with better strategies and comprehensive logging.
    
    Trains projection heads to map hidden states and truth embeddings
    to a shared space where semantic alignment can be measured.
    
    Args:
        config: Configuration object
        pairs: List of training pairs
    
    Returns:
        Trained ModelManager instance
    """
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

# ==============================================================
# Enhanced Feature Extraction and Analysis
# ==============================================================

class FeatureExtractor:
    """
    Comprehensive feature extraction for layer-wise semantic dynamics analysis.
    Extracts trajectory features across all model layers.
    """
    
    def __init__(self, model_manager: ModelManager):
        self.extractor = model_manager.extractor
        self.truth_encoder = model_manager.truth_encoder
        self.hidden_proj = model_manager.hidden_proj
        self.truth_proj = model_manager.truth_proj
    
    def extract_trajectory_features(self, text: str, truth: str) -> Dict[str, float]:
        """
        Extract comprehensive trajectory features from layer-wise alignments.
        
        Args:
            text: Input text to analyze
            truth: Ground truth statement
        
        Returns:
            Dictionary of trajectory features
        """
        with torch.no_grad():
            # Get hidden states for all layers
            hidden_states = self.extractor.get_hidden_states([text]).squeeze(0)  # [layers, hidden_dim]
            truth_embedding = self.truth_encoder.encode_batch([truth])  # [1, truth_dim]
            
            # Project to shared space
            hidden_projected = F.normalize(self.hidden_proj(hidden_states), p=2, dim=-1)  # [layers, shared_dim]
            truth_projected = F.normalize(self.truth_proj(truth_embedding), p=2, dim=-1)  # [1, shared_dim]
            
            # Compute layer-wise alignments
            alignments = []
            for layer_idx in range(hidden_projected.size(0)):
                layer_embedding = hidden_projected[layer_idx].unsqueeze(0)
                cos_sim = F.cosine_similarity(layer_embedding, truth_projected, dim=1)
                alignments.append(cos_sim.item())
            
            # Compute comprehensive trajectory metrics
            features = self._compute_trajectory_metrics(alignments, hidden_projected)
            
            return features
    
    def _compute_trajectory_metrics(self, alignments: List[float], 
                                  hidden_projected: torch.Tensor) -> Dict[str, float]:
        """
        Compute comprehensive trajectory metrics from layer-wise alignments.
        
        Features include:
        - Alignment statistics (final, mean, max)
        - Convergence metrics (which layer reaches peak)
        - Stability metrics (variance in late layers)
        - Change metrics (gain from first to last)
        - Dynamics metrics (velocity and acceleration)
        - Oscillation metrics (direction changes)
        """
        # Basic alignment metrics
        final_alignment = alignments[-1] if alignments else 0
        mean_alignment = np.mean(alignments) if alignments else 0
        max_alignment = np.max(alignments) if alignments else 0
        
        # Convergence metrics
        convergence_layer = int(np.argmax(alignments)) if alignments else 0
        
        # Stability metrics (variance in last 3 layers)
        stability = float(np.std(alignments[-3:])) if len(alignments) >= 3 else float(np.std(alignments)) if alignments else 0
        
        # Change metrics
        alignment_gain = float(alignments[-1] - alignments[0]) if len(alignments) > 1 else 0
        
        # Velocity and acceleration (change in embeddings)
        if hidden_projected.size(0) > 1:
            deltas = hidden_projected[1:] - hidden_projected[:-1]
            velocities = torch.norm(deltas, dim=1).cpu().numpy()
            mean_velocity = float(np.mean(velocities)) if len(velocities) > 0 else 0.0
            
            if len(deltas) > 2:
                accel_similarity = F.cosine_similarity(deltas[:-1], deltas[1:], dim=1)
                mean_acceleration = float(accel_similarity.mean().item())
            else:
                mean_acceleration = 0.0
        else:
            mean_velocity = 0.0
            mean_acceleration = 0.0
        
        # Oscillation metrics (direction changes)
        if len(alignments) > 2:
            second_derivative = np.diff(np.sign(np.diff(alignments)))
            oscillation_count = int(np.sum(second_derivative != 0))
        else:
            oscillation_count = 0
        
        return {
            # Alignment features
            'final_alignment': float(final_alignment),
            'mean_alignment': float(mean_alignment),
            'max_alignment': float(max_alignment),
            
            # Convergence features
            'convergence_layer': int(convergence_layer),
            
            # Stability features
            'stability': float(stability),
            
            # Change features
            'alignment_gain': float(alignment_gain),
            
            # Dynamics features
            'mean_velocity': float(mean_velocity),
            'mean_acceleration': float(mean_acceleration),
            
            # Oscillation features
            'oscillation_count': int(oscillation_count),
        }

def analyze_layerwise_dynamics(pairs: List[Tuple[str, str, str]], 
                             model_manager: ModelManager) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Comprehensive layer-wise dynamics analysis across all samples.
    
    Args:
        pairs: List of (text, truth, label) tuples
        model_manager: Trained model manager
    
    Returns:
        Tuple of (results_df, trajectories_dict, layerwise_data_dict)
    """
    feature_extractor = FeatureExtractor(model_manager)
    
    results = []
    all_trajectories = {"factual": [], "hallucination": []}
    layerwise_data = {"factual": [], "hallucination": []}
    
    logger.log("Starting comprehensive layer-wise dynamics analysis...")
    
    for text, truth, label in tqdm(pairs, desc="Analyzing Dynamics"):
        try:
            # Extract features
            features = feature_extractor.extract_trajectory_features(text, truth)
            
            # Get raw alignments for trajectory analysis
            with torch.no_grad():
                hidden_states = model_manager.extractor.get_hidden_states([text]).squeeze(0)
                truth_embedding = model_manager.truth_encoder.encode_batch([truth])
                
                hidden_projected = F.normalize(model_manager.hidden_proj(hidden_states), p=2, dim=-1)
                truth_projected = F.normalize(model_manager.truth_proj(truth_embedding), p=2, dim=-1)
                
                alignments = []
                for layer_idx in range(hidden_projected.size(0)):
                    layer_embedding = hidden_projected[layer_idx].unsqueeze(0)
                    cos_sim = F.cosine_similarity(layer_embedding, truth_projected, dim=1)
                    alignments.append(float(cos_sim.item()))
            
            # Store results
            result = {
                "label": label,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "truth": truth[:100] + "..." if len(truth) > 100 else truth,
                **features
            }
            
            results.append(result)
            all_trajectories[label].append(alignments)
            layerwise_data[label].append(alignments)
            
        except Exception as e:
            logger.log(f"Error processing sample: {e}", "WARNING")
            continue
    
    df = pd.DataFrame(results)
    df.to_csv(dir_manager.results_dir / "enhanced_dynamics_results.csv", index=False)
    
    logger.log(f"Analysis completed: {len(df)} samples processed")
    return df, all_trajectories, layerwise_data

# ==============================================================
# Enhanced Evaluation System
# ==============================================================

class ComprehensiveEvaluator:
    """
    Comprehensive evaluation with multiple strategies.
    Supports supervised, unsupervised, and hybrid evaluation modes.
    """
    
    def __init__(self, config: LayerwiseSemanticDynamicsConfig):
        self.config = config
    
    def compute_composite_score(self, metrics: Dict[str, float]) -> float:
        """
        Compute weighted composite detection score from multiple metrics.
        
        Args:
            metrics: Dictionary of metric values
        
        Returns:
            Composite score between 0 and 1
        """
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in self.config.composite_score_weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            score /= total_weight
        
        return float(score)
    
    def evaluate_supervised(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive supervised evaluation using multiple classifiers.
        
        Args:
            df: DataFrame with features and labels
        
        Returns:
            Dictionary of results for each classifier
        """
        logger.log("Starting supervised evaluation...")
        
        # Prepare features and labels
        feature_columns = [
            'final_alignment', 'mean_alignment', 'max_alignment', 'convergence_layer',
            'stability', 'alignment_gain', 'mean_velocity', 'mean_acceleration',
            'oscillation_count'
        ]
        
        # Filter available features
        available_features = [col for col in feature_columns if col in df.columns]
        X = df[available_features].values
        
        # Convert labels properly
        y = np.array([1 if label == 'factual' else 0 for label in df['label']])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define classifiers
        classifiers = {
            'LogisticRegression': LogisticRegression(random_state=SEED, max_iter=1000),
            'RandomForest': RandomForestClassifier(random_state=SEED, n_estimators=100),
            'GradientBoosting': GradientBoostingClassifier(random_state=SEED, n_estimators=100),
        }
        
        results = {}
        
        for clf_name, clf in classifiers.items():
            logger.log(f"Training {clf_name}...")
            
            # Train classifier
            clf.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = clf.predict(X_test_scaled)
            y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
            
            # Comprehensive metrics
            metrics = self._compute_comprehensive_metrics(y_test, y_pred, y_pred_proba)
            
            # Store test data for visualization
            metrics['y_true'] = [int(y) for y in y_test.tolist()]
            metrics['y_pred'] = [int(y) for y in y_pred.tolist()]
            metrics['y_pred_proba'] = [float(y) for y in y_pred_proba.tolist()]
            
            # Cross-validation scores
            try:
                cv_scores = cross_val_score(clf, X_train_scaled, y_train, 
                                          cv=min(self.config.cross_validation_folds, 5), 
                                          scoring='f1')
                metrics['cv_f1_mean'] = float(cv_scores.mean())
                metrics['cv_f1_std'] = float(cv_scores.std())
            except:
                metrics['cv_f1_mean'] = 0.0
                metrics['cv_f1_std'] = 0.0
            
            # Composite score
            metrics['composite_score'] = self.compute_composite_score(metrics)
            
            results[clf_name] = metrics
            
            logger.log(f"{clf_name} - F1: {metrics['f1']:.4f}, AUC-ROC: {metrics['auroc']:.4f}, "
                      f"Composite: {metrics['composite_score']:.4f}")
        
        return results
    
    def evaluate_unsupervised(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Unsupervised evaluation using clustering and anomaly detection.
        
        Args:
            df: DataFrame with features
        
        Returns:
            Dictionary of unsupervised metrics
        """
        logger.log("Starting unsupervised evaluation...")
        
        # Prepare features for clustering
        feature_columns = [
            'final_alignment', 'mean_alignment', 'stability', 'alignment_gain',
            'mean_velocity', 'mean_acceleration'
        ]
        
        available_features = [col for col in feature_columns if col in df.columns]
        X = df[available_features].values
        
        # Handle NaN values
        X = np.nan_to_num(X)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results = {}
        
        # K-means clustering
        try:
            kmeans = KMeans(n_clusters=2, random_state=SEED)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Evaluate clustering against ground truth
            true_labels = np.array([1 if label == 'factual' else 0 for label in df['label']])
            
            # Try both cluster mappings
            accuracy1 = accuracy_score(true_labels, cluster_labels)
            accuracy2 = accuracy_score(true_labels, 1 - cluster_labels)
            clustering_accuracy = float(max(accuracy1, accuracy2))
            
            results['clustering_accuracy'] = clustering_accuracy
        except Exception as e:
            logger.log(f"Clustering failed: {e}", "WARNING")
            results['clustering_accuracy'] = 0.0
        
        # Gaussian Mixture Model for anomaly detection
        try:
            gmm = GaussianMixture(n_components=2, random_state=SEED)
            gmm.fit(X_scaled)
            anomaly_scores = gmm.score_samples(X_scaled)
            
            results['anomaly_scores_mean'] = float(np.mean(anomaly_scores))
            results['anomaly_scores_std'] = float(np.std(anomaly_scores))
        except Exception as e:
            logger.log(f"GMM failed: {e}", "WARNING")
            results['anomaly_scores_mean'] = 0.0
            results['anomaly_scores_std'] = 0.0
        
        logger.log(f"Unsupervised analysis - Clustering accuracy: {results.get('clustering_accuracy', 0.0):.4f}")
        
        return results
    
    def evaluate_hybrid(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Hybrid evaluation combining supervised and unsupervised approaches.
        
        Args:
            df: DataFrame with features and labels
        
        Returns:
            Dictionary of hybrid results
        """
        supervised_results = self.evaluate_supervised(df)
        unsupervised_results = self.evaluate_unsupervised(df)
        
        # Combine results
        hybrid_results = {
            'supervised': supervised_results,
            'unsupervised': unsupervised_results,
            'hybrid_metrics': {}
        }
        
        # Compute hybrid metrics
        if supervised_results:
            best_supervised_score = float(max(
                result['composite_score'] for result in supervised_results.values()
            ))
            hybrid_results['hybrid_metrics']['best_supervised_score'] = best_supervised_score
            
            # Overall hybrid score
            hybrid_results['hybrid_metrics']['overall_hybrid_score'] = float(
                0.7 * best_supervised_score +
                0.3 * unsupervised_results.get('clustering_accuracy', 0.0)
            )
        
        return hybrid_results
    
    def _compute_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Includes: precision, recall, F1, F2, accuracy, specificity,
        AUC-ROC, AUC-PR, MCC, Cohen's Kappa, confusion matrix
        """
        try:
            # Basic metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            accuracy = accuracy_score(y_true, y_pred)
            
            # Confusion matrix components
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Advanced metrics
            auroc = roc_auc_score(y_true, y_pred_proba)
            
            # Precision-recall curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
            prauc = auc(recall_curve, precision_curve)
            
            # Additional metrics
            mcc = matthews_corrcoef(y_true, y_pred)
            kappa = cohen_kappa_score(y_true, y_pred)
            
            # F2 score (emphasizes recall)
            f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
            
            return {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'f2': float(f2),
                'accuracy': float(accuracy),
                'specificity': float(specificity),
                'auroc': float(auroc),
                'prauc': float(prauc),
                'mcc': float(mcc),
                'kappa': float(kappa),
                'confusion_matrix': {
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'true_positives': int(tp)
                }
            }
        except Exception as e:
            logger.log(f"Error computing metrics: {e}", "WARNING")
            # Return default metrics
            return {
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'f2': 0.0,
                'accuracy': 0.0, 'specificity': 0.0, 'auroc': 0.0, 'prauc': 0.0,
                'mcc': 0.0, 'kappa': 0.0,
                'confusion_matrix': {'true_negatives': 0, 'false_positives': 0, 
                                   'false_negatives': 0, 'true_positives': 0}
            }

# ==============================================================
# Statistical Analysis
# ==============================================================

class StatisticalAnalyzer:
    """
    Enhanced statistical analysis for layer-wise semantic dynamics.
    Performs t-tests, effect size calculations, and layer-wise significance analysis.
    """
    
    def __init__(self):
        self.alpha = 0.05
    
    def comprehensive_analysis(self, df: pd.DataFrame, layerwise_data: Dict) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis.
        
        Args:
            df: DataFrame with features and labels
            layerwise_data: Dictionary with layer-wise trajectory data
        
        Returns:
            Dictionary of statistical analysis results
        """
        logger.log("Performing comprehensive statistical analysis...")
        
        results = {
            'group_comparisons': {},
            'layerwise_analysis': {},
            'correlation_analysis': {},
            'effect_sizes': {}
        }
        
        # Group comparisons
        results['group_comparisons'] = self._compare_groups(df)
        
        # Layer-wise analysis
        results['layerwise_analysis'] = self._analyze_layerwise_significance(layerwise_data)
        
        # Summary statistics
        results['summary'] = self._generate_summary(df, results)
        
        return results
    
    def _compare_groups(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare factual vs hallucination groups using t-tests.
        
        Args:
            df: DataFrame with features and labels
        
        Returns:
            Dictionary of comparison statistics for each feature
        """
        comparisons = {}
        
        # Select numeric columns for comparison
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            factual_vals = df[df['label'] == 'factual'][col].dropna()
            hallucination_vals = df[df['label'] == 'hallucination'][col].dropna()
            
            if len(factual_vals) > 1 and len(hallucination_vals) > 1:
                try:
                    # T-test
                    t_stat, p_value = ttest_ind(factual_vals, hallucination_vals)
                    
                    # Effect size (Cohen's d) with safe division
                    pooled_std = np.sqrt((factual_vals.std()**2 + hallucination_vals.std()**2) / 2)
                    if pooled_std > 0:
                        cohens_d = (factual_vals.mean() - hallucination_vals.mean()) / pooled_std
                    else:
                        cohens_d = 0.0
                    
                    comparisons[col] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'cohens_d': float(cohens_d),
                        'significant': bool(p_value < self.alpha),
                        'factual_mean': float(factual_vals.mean()),
                        'hallucination_mean': float(hallucination_vals.mean()),
                    }
                except Exception as e:
                    logger.log(f"Error comparing {col}: {e}", "WARNING")
                    continue
        
        return comparisons
    
    def _analyze_layerwise_significance(self, layerwise_data: Dict) -> Dict[str, Any]:
        """
        Analyze statistical significance across layers.
        
        Args:
            layerwise_data: Dictionary with layer-wise trajectory data
        
        Returns:
            Dictionary of layer-wise significance results
        """
        layer_results = {}
        
        if 'factual' in layerwise_data and 'hallucination' in layerwise_data:
            factual_trajs = layerwise_data['factual']
            hallucination_trajs = layerwise_data['hallucination']
            
            if factual_trajs and hallucination_trajs:
                min_layers = min(len(factual_trajs[0]), len(hallucination_trajs[0]))
                
                p_values = []
                effect_sizes = []
                
                for layer in range(min_layers):
                    try:
                        factual_vals = [traj[layer] for traj in factual_trajs]
                        hallucination_vals = [traj[layer] for traj in hallucination_trajs]
                        
                        t_stat, p_val = ttest_ind(factual_vals, hallucination_vals)
                        pooled_std = np.sqrt((np.std(factual_vals)**2 + np.std(hallucination_vals)**2) / 2)
                        
                        if pooled_std > 0:
                            cohens_d = (np.mean(factual_vals) - np.mean(hallucination_vals)) / pooled_std
                        else:
                            cohens_d = 0.0
                        
                        p_values.append(float(p_val))
                        effect_sizes.append(float(cohens_d))
                    except:
                        p_values.append(1.0)
                        effect_sizes.append(0.0)
                
                layer_results = {
                    'p_values': p_values,
                    'effect_sizes': effect_sizes,
                    'significant_layers': int(np.sum(np.array(p_values) < self.alpha)) if p_values else 0,
                }
        
        return layer_results
    
    def _generate_summary(self, df: pd.DataFrame, results: Dict) -> Dict[str, Any]:
        """
        Generate statistical summary.
        
        Args:
            df: DataFrame with results
            results: Dictionary of analysis results
        
        Returns:
            Dictionary of summary statistics
        """
        group_comparisons = results['group_comparisons']
        
        significant_metrics = [
            metric for metric, stats in group_comparisons.items()
            if stats.get('significant', False)
        ]
        
        return {
            'total_samples': int(len(df)),
            'factual_samples': int(len(df[df['label'] == 'factual'])),
            'hallucination_samples': int(len(df[df['label'] == 'hallucination'])),
            'significant_metrics_count': int(len(significant_metrics)),
        }

# ==============================================================
# Visualization System
# ==============================================================

class VisualizationEngine:
    """
    Comprehensive visualization system for layer-wise semantic dynamics analysis.
    Generates plots for metrics comparison, ROC curves, and performance analysis.
    """
    
    def __init__(self):
        self.style_config = {
            'factual_color': 'green',
            'hallucination_color': 'red',
            'neutral_color': 'blue',
            'cmap': 'RdYlGn',
            'figsize_large': (16, 12),
            'figsize_medium': (12, 8),
            'figsize_small': (8, 6)
        }
    
    def plot_comprehensive_metrics(self, results: Dict[str, Any]):
        """
        Plot comprehensive evaluation metrics across all classifiers.
        
        Args:
            results: Dictionary of evaluation results
        """
        logger.log("Generating comprehensive metrics plots...")
        
        if 'supervised' not in results:
            logger.log("No supervised results to plot", "WARNING")
            return
        
        supervised_results = results['supervised']
        
        if not supervised_results:
            logger.log("No supervised results to plot", "WARNING")
            return
        
        # Create comprehensive figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=self.style_config['figsize_medium'])
        
        # Extract method names and metrics
        methods = list(supervised_results.keys())
        metrics = ['precision', 'recall', 'f1', 'auroc']
        metric_names = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        
        # Plot 1: Main metrics comparison
        metric_values = {metric: [supervised_results[method][metric] for method in methods] 
                        for metric in metrics}
        
        x = np.arange(len(methods))
        width = 0.2
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            axes[0, 0].bar(x + i * width, metric_values[metric], width, 
                          label=metric_name, alpha=0.8)
        
        axes[0, 0].set_xlabel('Methods')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Comprehensive Metrics Comparison')
        axes[0, 0].set_xticks(x + width * 1.5)
        axes[0, 0].set_xticklabels(methods, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: ROC Curves
        for method_name, result in supervised_results.items():
            if 'y_true' in result and 'y_pred_proba' in result:
                fpr, tpr, _ = roc_curve(result['y_true'], result['y_pred_proba'])
                axes[0, 1].plot(fpr, tpr, label=f'{method_name} (AUC = {result["auroc"]:.3f})', 
                               linewidth=2)
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Composite Scores
        composite_scores = [result['composite_score'] for result in supervised_results.values()]
        axes[1, 0].bar(methods, composite_scores, color=['skyblue' for _ in methods])
        axes[1, 0].set_xlabel('Methods')
        axes[1, 0].set_ylabel('Composite Score')
        axes[1, 0].set_title('Composite Detection Scores')
        axes[1, 0].set_xticks(range(len(methods)))
        axes[1, 0].set_xticklabels(methods, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value annotations on bars
        for i, v in enumerate(composite_scores):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Plot 4: Cross-Validation Performance
        if 'cv_f1_mean' in list(supervised_results.values())[0]:
            cv_means = [result['cv_f1_mean'] for result in supervised_results.values()]
            cv_stds = [result['cv_f1_std'] for result in supervised_results.values()]
            
            axes[1, 1].bar(methods, cv_means, yerr=cv_stds, capsize=5, alpha=0.7,
                          color=['lightgreen' for _ in methods])
            axes[1, 1].set_xlabel('Methods')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].set_title('Cross-Validation Performance')
            axes[1, 1].set_xticks(range(len(methods)))
            axes[1, 1].set_xticklabels(methods, rotation=45, ha='right')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(dir_manager.get_plot_path("comprehensive_metrics"), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.log("Comprehensive metrics plots generated successfully")

# ==============================================================
# Main Execution and Orchestration
# ==============================================================

class AnalysisOrchestrator:
    """
    Enhanced orchestrator with better error handling and diagnostics.
    Manages the complete analysis pipeline from data loading to final reporting.
    """
    
    def __init__(self, config: LayerwiseSemanticDynamicsConfig):
        self.config = config
        self.data_manager = DataManager(config)
        self.evaluator = ComprehensiveEvaluator(config)
        self.visualizer = VisualizationEngine()
        self.stat_analyzer = StatisticalAnalyzer()
        
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive layer-wise semantic dynamics analysis.
        
        Complete pipeline:
        1. Build dataset from multiple sources
        2. Train projection heads with contrastive learning
        3. Analyze layer-wise dynamics and extract features
        4. Perform statistical analysis
        5. Evaluate using supervised/unsupervised methods
        6. Generate visualizations
        7. Create comprehensive report
        
        Returns:
            Dictionary containing final analysis report
        """
        logger.log("=" * 80)
        logger.log("STARTING LAYER-WISE SEMANTIC DYNAMICS ANALYSIS")
        logger.log("=" * 80)
        
        try:
            # Step 1: Build dataset
            logger.log("\n[STEP 1/7] Building dataset...")
            pairs = self.data_manager.build_dataset()
            
            if not pairs:
                logger.log("ERROR: No pairs generated. Exiting.", "ERROR")
                return {"status": "error", "message": "No data pairs generated"}
            
            # Step 2: Enhanced training
            logger.log("\n[STEP 2/7] Running enhanced training...")
            model_manager = train_projection_heads(self.config, pairs)
            
            # Step 3: Analyze dynamics
            logger.log("\n[STEP 3/7] Analyzing layer-wise dynamics...")
            df, all_trajectories, layerwise_data = analyze_layerwise_dynamics(pairs, model_manager)
            
            if df.empty:
                logger.log("ERROR: No analysis results generated. Exiting.", "ERROR")
                return {"status": "error", "message": "Analysis produced no results"}
            
            # Step 4: Statistical analysis
            logger.log("\n[STEP 4/7] Performing statistical analysis...")
            stats_summary = self.stat_analyzer.comprehensive_analysis(df, layerwise_data)
            
            # Step 5: Evaluation based on mode
            logger.log(f"\n[STEP 5/7] Running {self.config.mode.value} evaluation...")
            
            if self.config.mode == OperationMode.SUPERVISED:
                evaluation_results = self.evaluator.evaluate_supervised(df)
            elif self.config.mode == OperationMode.UNSUPERVISED:
                evaluation_results = self.evaluator.evaluate_unsupervised(df)
            else:  # HYBRID
                evaluation_results = self.evaluator.evaluate_hybrid(df)
            
            # Step 6: Visualization
            logger.log("\n[STEP 6/7] Generating visualizations...")
            self.visualizer.plot_comprehensive_metrics(evaluation_results)
            
            # Step 7: Generate comprehensive report
            logger.log("\n[STEP 7/7] Generating final report...")
            final_report = self._generate_final_report(
                df, evaluation_results, stats_summary
            )
            
            # Save all results
            self._save_results(df, evaluation_results, stats_summary, final_report)
            
            logger.log("=" * 80)
            logger.log("COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY")
            logger.log("=" * 80)
            
            return final_report
            
        except Exception as e:
            logger.log(f"ERROR in comprehensive analysis: {e}", "ERROR")
            import traceback
            logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")
            return {"status": "error", "message": str(e)}
    
    def _generate_final_report(self, df: pd.DataFrame, evaluation_results: Dict,
                             stats_summary: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive final report.
        
        Args:
            df: Results DataFrame
            evaluation_results: Evaluation metrics
            stats_summary: Statistical analysis summary
        
        Returns:
            Comprehensive report dictionary
        """
        
        report = {
            'execution_summary': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'config': {
                    'model_name': self.config.model_name,
                    'mode': self.config.mode.value,
                    'num_pairs': self.config.num_pairs
                },
                'device': DEVICE,
                'total_samples': int(len(df))
            },
            'dataset_statistics': {
                'factual_samples': int(len(df[df['label'] == 'factual'])),
                'hallucination_samples': int(len(df[df['label'] == 'hallucination'])),
                'class_balance': float(len(df[df['label'] == 'factual']) / len(df)) if len(df) > 0 else 0.0
            },
            'evaluation_results': evaluation_results,
            'statistical_summary': stats_summary['summary'],
            'key_findings': {},
            'recommendations': []
        }
        
        # Extract key findings
        if 'supervised' in evaluation_results:
            supervised_results = evaluation_results['supervised']
            if supervised_results:
                best_method = max(supervised_results.items(), 
                                key=lambda x: x[1]['composite_score'])
                
                report['key_findings']['best_method'] = best_method[0]
                report['key_findings']['best_composite_score'] = float(best_method[1]['composite_score'])
                report['key_findings']['best_f1_score'] = float(best_method[1]['f1'])
                
                # Detection quality assessment
                composite_score = best_method[1]['composite_score']
                if composite_score >= 0.9:
                    detection_class = "EXCELLENT"
                elif composite_score >= 0.8:
                    detection_class = "VERY_GOOD" 
                elif composite_score >= 0.7:
                    detection_class = "GOOD"
                else:
                    detection_class = "MODERATE"
                    
                report['key_findings']['detection_quality'] = detection_class
        
        # Statistical findings
        stats_summary_data = stats_summary['summary']
        report['key_findings']['significant_metrics'] = int(stats_summary_data['significant_metrics_count'])
        
        # Generate recommendations
        self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: Dict[str, Any]):
        """
        Generate actionable recommendations based on analysis results.
        
        Args:
            report: Report dictionary to add recommendations to
        """
        recommendations = []
        
        # Based on detection quality
        detection_quality = report['key_findings'].get('detection_quality', 'MODERATE')
        if detection_quality == "EXCELLENT":
            recommendations.append("✓ Ready for production deployment in critical applications")
        elif detection_quality == "VERY_GOOD":
            recommendations.append("✓ Suitable for production deployment in most applications")
        elif detection_quality == "GOOD":
            recommendations.append("✓ Suitable for deployment with monitoring and validation")
        else:
            recommendations.append("⚠ Consider further optimization before production deployment")
        
        # Based on statistical findings
        sig_metrics = report['key_findings'].get('significant_metrics', 0)
        if sig_metrics >= 5:
            recommendations.append("✓ Strong statistical foundation with multiple significant metrics")
        elif sig_metrics >= 3:
            recommendations.append("✓ Good statistical foundation")
        else:
            recommendations.append("⚠ Limited statistical significance - consider more data")
        
        # Based on sample size
        total_samples = report['execution_summary']['total_samples']
        if total_samples < 100:
            recommendations.append("⚠ Small sample size - collect more data for robust results")
        elif total_samples < 500:
            recommendations.append("✓ Adequate sample size for initial validation")
        else:
            recommendations.append("✓ Good sample size for reliable analysis")
        
        report['recommendations'] = recommendations
    
    def _save_results(self, df: pd.DataFrame, evaluation_results: Dict,
                     stats_summary: Dict, final_report: Dict):
        """
        Save all results to files with proper JSON serialization.
        
        Args:
            df: Results DataFrame
            evaluation_results: Evaluation metrics
            stats_summary: Statistical summary
            final_report: Final report
        """
        try:
            # Save dataframe
            df.to_csv(dir_manager.results_dir / "final_analysis_results.csv", index=False)
            logger.log("Saved analysis results CSV")
            
            # Save evaluation results with type conversion
            safe_json_dump(
                evaluation_results,
                dir_manager.get_result_path("evaluation_results")
            )
            logger.log("Saved evaluation results JSON")
            
            # Save statistical summary with type conversion
            safe_json_dump(
                stats_summary,
                dir_manager.get_result_path("statistical_summary")
            )
            logger.log("Saved statistical summary JSON")
            
            # Save final report with type conversion
            safe_json_dump(
                final_report,
                dir_manager.get_result_path("final_report")
            )
            logger.log("Saved final report JSON")
            
            logger.log(f"All results saved to: {dir_manager.results_dir}")
            
        except Exception as e:
            logger.log(f"Error saving results: {e}", "ERROR")
            import traceback
            logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")

# ==============================================================
# Usage Examples and Configuration
# ==============================================================

def create_enhanced_config() -> LayerwiseSemanticDynamicsConfig:
    """
    Create enhanced detection configuration with optimized parameters.
    
    Returns:
        Configured LayerwiseSemanticDynamicsConfig object
    """
    return LayerwiseSemanticDynamicsConfig(
        model_name="gpt2",
        truth_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        num_pairs=1000,
        epochs=10,
        batch_size=8,
        learning_rate=5e-5,
        margin=0.5,
        mode=OperationMode.HYBRID,
        use_pretrained=False,
        enable_ensemble=True,
        composite_score_weights={
            'f1': 0.25,
            'auroc': 0.20,
            'precision': 0.15,
            'recall': 0.15,
            'specificity': 0.10,
            'mcc': 0.15
        }
    )

def main():
    """
    Main execution function.
    
    Orchestrates the complete layer-wise semantic dynamics analysis pipeline.
    """
    
    print("\n" + "="*80)
    print("LAYER-WISE SEMANTIC DYNAMICS ANALYSIS SYSTEM")
    print("Hallucination Detection via Semantic Trajectory Analysis")
    print("="*80 + "\n")
    
    # Use enhanced configuration
    config = create_enhanced_config()
    
    print(f"Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Mode: {config.mode.value}")
    print(f"  Samples: {config.num_pairs}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Device: {DEVICE}\n")
    
    # Create and run orchestrator
    orchestrator = AnalysisOrchestrator(config)
    final_report = orchestrator.run_comprehensive_analysis()
    
    # Print summary
    if final_report and 'status' not in final_report or final_report.get('status') != 'error':
        print("\n" + "="*80)
        print("LAYER-WISE SEMANTIC DYNAMICS - ANALYSIS COMPLETE")
        print("="*80)
        
        print(f"\nEXECUTION SUMMARY:")
        print(f"  Total samples analyzed: {final_report['execution_summary']['total_samples']}")
        print(f"  Operation mode: {config.mode.value}")
        print(f"  Device: {final_report['execution_summary']['device']}")
        print(f"  Timestamp: {final_report['execution_summary']['timestamp']}")
        
        if 'dataset_statistics' in final_report:
            ds = final_report['dataset_statistics']
            print(f"\nDATASET STATISTICS:")
            print(f"  Factual samples: {ds['factual_samples']}")
            print(f"  Hallucination samples: {ds['hallucination_samples']}")
            print(f"  Class balance: {ds['class_balance']:.2%}")
        
        if 'key_findings' in final_report:
            kf = final_report['key_findings']
            print(f"\nKEY FINDINGS:")
            print(f"  Best method: {kf.get('best_method', 'N/A')}")
            print(f"  Composite score: {kf.get('best_composite_score', 0):.4f}")
            print(f"  F1 score: {kf.get('best_f1_score', 0):.4f}")
            print(f"  Detection quality: {kf.get('detection_quality', 'N/A')}")
            print(f"  Significant metrics: {kf.get('significant_metrics', 0)}")
        
        print(f"\nRECOMMENDATIONS:")
        for rec in final_report.get('recommendations', []):
            print(f"  {rec}")
        
        print(f"\nRESULTS SAVED TO: {dir_manager.base_dir}")
        print("  - Models: models/")
        print("  - Plots: plots/")
        print("  - Results: results/")
        print("  - Logs: execution.log")
        print("="*80 + "\n")
        
        return final_report
    else:
        print("\n" + "="*80)
        print("ANALYSIS FAILED")
        print("="*80)
        print(f"Error: {final_report.get('message', 'Unknown error')}")
        print("Check the logs for details.")
        print("="*80 + "\n")
        return None

if __name__ == "__main__":
    main()