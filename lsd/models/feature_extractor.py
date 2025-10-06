import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Any
from ..utils.helpers import logger, dir_manager

class FeatureExtractor:
    """Comprehensive feature extraction for layer-wise semantic dynamics analysis."""
    
    def __init__(self, model_manager):
        self.extractor = model_manager.extractor
        self.truth_encoder = model_manager.truth_encoder
        self.hidden_proj = model_manager.hidden_proj
        self.truth_proj = model_manager.truth_proj
    
    def extract_trajectory_features(self, text: str, truth: str) -> Dict[str, float]:
        """Extract comprehensive trajectory features from layer-wise alignments."""
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
        """Compute comprehensive trajectory metrics from layer-wise alignments."""
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
                             model_manager) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Comprehensive layer-wise dynamics analysis across all samples."""
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