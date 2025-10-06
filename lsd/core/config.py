from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

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