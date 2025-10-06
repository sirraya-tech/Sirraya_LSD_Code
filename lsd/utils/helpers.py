import os
import math
import random
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import torch

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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