import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import numpy as np

from ..utils.helpers import DEVICE, logger

def mean_pool_hidden(hidden, attn_mask):
    """Mean pool hidden states using attention mask."""
    mask = attn_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

class HiddenStatesExtractor:
    """Extracts hidden states from all layers of a language model."""
    
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
        """Extract hidden states from all layers for given texts."""
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
    """Encodes truth statements using sentence transformers."""
    
    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 device: str = DEVICE, use_cache: bool = True):
        logger.log(f"Loading truth encoder: {name}")
        self.model = SentenceTransformer(name).to(device)
        self.use_cache = use_cache
        self._cache = {}
        logger.log(f"Initialized TruthEncoder: {name}")
        
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Encode texts with caching and L2 normalization."""
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
    """Build enhanced projection heads with deep architecture."""
    
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

def enhanced_contrastive_loss(cos_sim: torch.Tensor, labels: torch.Tensor, 
                            margin: float = 0.5) -> torch.Tensor:
    """Enhanced contrastive loss with better gradient flow."""
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    # Positive pairs - we want high similarity (close to 1)
    pos_loss = (1 - cos_sim[pos_mask]).pow(2).mean() if pos_mask.any() else 0
    
    # Negative pairs - we want low similarity (<= -margin)
    neg_loss = F.relu(cos_sim[neg_mask] + margin).pow(2).mean() if neg_mask.any() else 0
    
    # Combine losses with equal weighting
    total_loss = 0.5 * pos_loss + 0.5 * neg_loss
    
    return total_loss