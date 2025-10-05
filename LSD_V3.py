# ==============================================================
# Layer-wise Semantic Dynamics (LSD) - Fixed Implementation
# Proper System Integration
# ==============================================================

import os
import math
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from contextlib import contextmanager
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from scipy.stats import ttest_ind, mannwhitneyu, pearsonr
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')

# ==============================================================
# Enhanced Configuration
# ==============================================================

@dataclass
class LSDConfig:
    """Robust configuration for Layer-wise Semantic Dynamics analysis"""
    model_name: str = "gpt2"
    truth_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    shared_dim: int = 384
    batch_size: int = 4
    epochs: int = 15
    learning_rate: float = 5e-5
    margin: float = 0.4
    max_length: int = 128
    num_pairs: int = 500
    seed: int = 42
    datasets: List[str] = field(default_factory=lambda: ["synthetic"])
    early_stopping_patience: int = 3
    gradient_clip: float = 1.0
    weight_decay: float = 1e-5

# ==============================================================
# System Initialization
# ==============================================================

class SystemManager:
    """Complete system management with resource handling"""
    
    def __init__(self, config: LSDConfig):
        self.config = config
        self.device = self._setup_device()
        self.directories = self._setup_directories()
        self._set_seeds()
        
    def _setup_device(self) -> torch.device:
        """Setup device with fallback strategy"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        print(f"Using device: {device}")
        return device
    
    def _setup_directories(self) -> Dict[str, Path]:
        """Create organized directory structure"""
        base_dir = Path("lsd_robust_analysis")
        directories = {
            'base': base_dir,
            'models': base_dir / "models",
            'plots': base_dir / "plots",
            'results': base_dir / "results",
            'logs': base_dir / "logs"
        }
        
        for dir_path in directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        return directories
    
    def _set_seeds(self):
        """Set all random seeds for reproducibility"""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

# ==============================================================
# Enhanced Logging
# ==============================================================

class EnhancedLogger:
    """Sophisticated logging with context management"""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self._log_queue = []
        
    def log(self, message: str, level: str = "INFO", print_message: bool = True):
        """Enhanced logging with levels and formatting"""
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] [{level:8}] {message}"
        
        self._log_queue.append(log_entry)
        
        if print_message:
            color_codes = {
                "INFO": "\033[94m",
                "SUCCESS": "\033[92m",
                "WARNING": "\033[93m",
                "ERROR": "\033[91m",
                "DEBUG": "\033[90m"
            }
            reset_code = "\033[0m"
            
            color = color_codes.get(level, "\033[0m")
            print(f"{color}{log_entry}{reset_code}")
    
    def flush(self):
        """Flush logs to file"""
        with open(self.log_file, "a", encoding="utf-8") as f:
            for entry in self._log_queue:
                f.write(entry + "\n")
        self._log_queue.clear()
    
    @contextmanager
    def task_context(self, task_name: str):
        """Context manager for task timing and status reporting"""
        start_time = pd.Timestamp.now()
        self.log(f"Starting: {task_name}", "INFO")
        
        try:
            yield
            duration = (pd.Timestamp.now() - start_time).total_seconds()
            self.log(f"Completed: {task_name} ({duration:.2f}s)", "SUCCESS")
        except Exception as e:
            duration = (pd.Timestamp.now() - start_time).total_seconds()
            self.log(f"Failed: {task_name} after {duration:.2f}s - {str(e)}", "ERROR")
            raise

# ==============================================================
# Optimized Model Components
# ==============================================================

class EfficientHuggingFaceExtractor:
    """Memory-efficient hidden state extraction with proper gradient isolation"""
    
    def __init__(self, model_name: str, device: torch.device, max_length: int):
        self.device = device
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            torch_dtype=torch.float32,
        ).to(device).eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
    
    @torch.no_grad()
    def get_hidden_states(self, texts: List[str]) -> torch.Tensor:
        """Efficient hidden state extraction with gradient isolation"""
        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_attention_mask=True
        ).to(self.device)
        
        outputs = self.model(**encodings, output_hidden_states=True)
        
        # Efficient mean pooling
        hidden_states = []
        attention_mask = encodings['attention_mask']
        
        for layer_hidden in outputs.hidden_states:
            mask_expanded = attention_mask.unsqueeze(-1).to(layer_hidden.dtype)
            sum_embeddings = torch.sum(layer_hidden * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask
            hidden_states.append(pooled)
        
        return torch.stack(hidden_states, dim=1)

class RobustTruthEncoder:
    """Robust truth encoding with proper gradient handling"""
    
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.model = SentenceTransformer(model_name, device=str(device))
        self.model.requires_grad_(False)
    
    @torch.no_grad()
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Encode with no gradient computation"""
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).to(self.device)
        return embeddings

# ==============================================================
# Advanced Projection Networks
# ==============================================================

class ResidualProjectionBlock(nn.Module):
    """Residual projection block with modern architecture"""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.shortcut(x)

class AdvancedProjectionHead(nn.Module):
    """Advanced projection head with residual connections"""
    
    def __init__(self, input_dim: int, output_dim: int, num_blocks: int = 2):
        super().__init__()
        
        blocks = []
        current_dim = input_dim
        
        for i in range(num_blocks):
            blocks.append(ResidualProjectionBlock(current_dim, output_dim))
            current_dim = output_dim
        
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        return F.normalize(self.blocks(x), p=2, dim=-1)

# ==============================================================
# Robust Dataset Management
# ==============================================================

class QualityDatasetBuilder:
    """Robust dataset builder with quality control"""
    
    def __init__(self, config: LSDConfig):
        self.config = config
    
    def build_quality_pairs(self) -> List[Tuple[str, str, str]]:
        """Build high-quality dataset pairs"""
        base_pairs = self._get_base_pairs()
        validated_pairs = self._validate_pairs(base_pairs)
        balanced_pairs = self._balance_dataset(validated_pairs)
        
        return balanced_pairs[:self.config.num_pairs]
    
    def _get_base_pairs(self) -> List[Tuple[str, str, str]]:
        """Get base pairs with comprehensive coverage"""
        return [
            # Factual pairs
            ("The Earth orbits the Sun.", "Earth revolves around the Sun once a year.", "factual"),
            ("Water boils at 100°C at sea level.", "Water boils at 100 degrees Celsius at standard pressure.", "factual"),
            ("Photosynthesis produces oxygen.", "Plants release oxygen during photosynthesis.", "factual"),
            ("The human body has 206 bones.", "An adult human skeleton has 206 bones.", "factual"),
            ("Shakespeare wrote Hamlet.", "William Shakespeare authored Hamlet.", "factual"),
            ("Mount Everest is the highest mountain.", "Mount Everest has the highest elevation above sea level.", "factual"),
            ("The capital of France is Paris.", "Paris is the capital city of France.", "factual"),
            
            # Hallucination pairs
            ("The Earth orbits the Moon.", "Earth orbits the Sun, not the Moon.", "hallucination"),
            ("Water boils at 50°C at sea level.", "Water boils at 100°C at sea level.", "hallucination"),
            ("Photosynthesis consumes oxygen.", "Photosynthesis produces oxygen.", "hallucination"),
            ("The human body has 300 bones.", "Adults have 206 bones, not 300.", "hallucination"),
            ("Shakespeare wrote Harry Potter.", "J.K. Rowling wrote Harry Potter.", "hallucination"),
            ("Mount Everest is in Europe.", "Mount Everest is in Asia, in the Himalayas.", "hallucination"),
            ("The capital of France is London.", "Paris is the capital of France, not London.", "hallucination"),

            # Factual pairs
          ("The Earth orbits the Sun.", "Earth revolves around the Sun once a year.", "factual"),
          ("Water boils at 100°C at sea level.", "Water boils at 100 degrees Celsius at standard atmospheric pressure.", "factual"),
          ("Photosynthesis produces oxygen.", "Plants release oxygen during photosynthesis.", "factual"),
          ("The human body has 206 bones.", "An adult human skeleton typically consists of 206 bones.", "factual"),
          ("Shakespeare wrote Hamlet.", "William Shakespeare is the author of the play Hamlet.", "factual"),
          ("The capital of France is Paris.", "Paris is the capital and largest city of France.", "factual"),
          ("Dolphins are mammals.", "Dolphins are marine mammals that breathe air.", "factual"),
          ("The Great Wall is in China.", "The Great Wall of China is a series of fortifications in northern China.", "factual"),
          ("Einstein developed relativity theory.", "Albert Einstein formulated the theory of relativity.", "factual"),
          ("DNA contains genetic information.", "Deoxyribonucleic acid carries genetic instructions.", "factual"),
          
          # Hallucination pairs
          ("The Earth orbits the Moon.", "Earth revolves around the Sun yearly.", "hallucination"),
          ("Water boils at 50°C at sea level.", "Water requires 100°C to boil at sea level.", "hallucination"),
          ("Photosynthesis consumes oxygen.", "Plants produce oxygen through photosynthesis.", "hallucination"),
          ("The human body has 300 bones.", "An adult human has 206 bones in their skeleton.", "hallucination"),
          ("Shakespeare wrote Harry Potter.", "J.K. Rowling wrote the Harry Potter series.", "hallucination"),
          ("The capital of France is London.", "Paris is the capital of France, not London.", "hallucination"),
          ("Dolphins are fish.", "Dolphins are mammals, not fish.", "hallucination"),
          ("The Great Wall is in Japan.", "The Great Wall is located in China.", "hallucination"),
          ("Newton developed relativity theory.", "Einstein, not Newton, developed relativity.", "hallucination"),
          ("RNA contains genetic information.", "DNA, not RNA, contains primary genetic information.", "hallucination"),
          
          # Additional factual pairs
          ("Gravity causes objects to fall.", "Objects fall due to gravitational attraction.", "factual"),
          ("Ice melts at 0°C.", "Ice changes to water at 0 degrees Celsius.", "factual"),
          ("The heart pumps blood.", "The heart circulates blood throughout the body.", "factual"),
          ("Python is a programming language.", "Python is widely used for software development.", "factual"),
          ("Mount Everest is the highest mountain.", "Mount Everest is Earth's highest mountain above sea level.", "factual"),
          
          # Additional hallucination pairs
          ("Gravity pushes objects upward.", "Gravity pulls objects downward, not upward.", "hallucination"),
          ("Ice melts at 100°C.", "Ice melts at 0°C, not 100°C.", "hallucination"),
          ("The lungs pump blood.", "The heart pumps blood, not the lungs.", "hallucination"),
          ("Python is a snake species only.", "Python is both a snake and programming language.", "hallucination"),
          ("Mount Everest is in Europe.", "Mount Everest is in Asia, specifically the Himalayas.", "hallucination"),

          ("The Moon orbits the Earth.", "Earth’s natural satellite is the Moon.", "factual"),
          ("The human brain controls the body.", "The brain regulates body functions and behavior.", "factual"),
          ("The Pacific Ocean is the largest ocean.", "The Pacific Ocean covers the largest surface area.", "factual"),
          ("Water freezes at 0°C.", "The freezing point of water is 0 degrees Celsius.", "factual"),
          ("The Sun is a star.", "The Sun is classified as a G-type main-sequence star.", "factual"),
          ("Plants produce oxygen.", "During photosynthesis, plants release oxygen.", "factual"),
          ("The capital of Japan is Tokyo.", "Tokyo is the capital city of Japan.", "factual"),
          ("Blood carries oxygen through the body.", "Red blood cells transport oxygen in the bloodstream.", "factual"),
          ("The Great Wall is in China.", "The Great Wall of China is located in northern China.", "factual"),
          ("The Earth revolves around the Sun once a year.", "Earth completes one orbit around the Sun every 365 days.", "factual"),
          ("Dogs are mammals.", "Dogs belong to the mammal class.", "factual"),
          ("Bats can fly.", "Bats are flying mammals capable of sustained flight.", "factual"),
          ("Albert Einstein developed the theory of relativity.", "Einstein formulated the special and general theories of relativity.", "factual"),
          ("A square has four equal sides.", "All sides of a square are equal in length.", "factual"),
          ("The chemical formula for water is H₂O.", "Water is composed of two hydrogen atoms and one oxygen atom.", "factual"),
          ("Bees produce honey.", "Bees create honey from nectar collected from flowers.", "factual"),
          ("Light travels faster than sound.", "The speed of light exceeds the speed of sound.", "factual"),
          ("The Eiffel Tower is in Paris.", "The Eiffel Tower is located in Paris, France.", "factual"),
          ("Mount Everest is the tallest mountain on Earth.", "Mount Everest has the highest elevation above sea level.", "factual"),
          ("The human skeleton has 206 bones.", "An adult human body contains 206 bones.", "factual"),
          ("The heart has four chambers.", "The human heart is made up of four chambers.", "factual"),
          ("The Amazon River is in South America.", "The Amazon River flows through South America.", "factual"),
          ("Electric current is measured in amperes.", "The SI unit of electric current is the ampere.", "factual"),
          ("The human body temperature averages 37°C.", "The average normal body temperature is around 37 degrees Celsius.", "factual"),
          ("The Statue of Liberty is in New York City.", "The Statue of Liberty stands on Liberty Island in New York.", "factual"),
          ("Carbon dioxide is absorbed by plants.", "Plants use carbon dioxide during photosynthesis.", "factual"),
          ("The speed of light is about 300,000 km/s.", "Light travels at approximately 300,000 kilometers per second.", "factual"),
          ("Penguins live in the Southern Hemisphere.", "Most penguin species are found in the Southern Hemisphere.", "factual"),
          ("Venus is the second planet from the Sun.", "Venus orbits as the second planet from the Sun.", "factual"),
          ("Shakespeare wrote Macbeth.", "William Shakespeare is the author of Macbeth.", "factual"),
          ("Photosynthesis requires sunlight.", "Plants need sunlight to perform photosynthesis.", "factual"),
          ("The human heart pumps blood.", "The heart circulates blood throughout the body.", "factual"),
          ("The boiling point of water is 100°C.", "Water boils at 100 degrees Celsius at sea level.", "factual"),
          ("Oxygen is essential for human respiration.", "Humans need oxygen to breathe.", "factual"),
          ("The Sahara is the largest hot desert on Earth.", "The Sahara Desert is the world’s largest hot desert.", "factual"),
          ("Mars is known as the Red Planet.", "Mars appears red due to iron oxide on its surface.", "factual"),
          ("DNA carries genetic information.", "Genetic instructions are encoded in DNA molecules.", "factual"),
          ("Saturn has visible rings.", "Saturn’s ring system is made of ice and rock particles.", "factual"),
          ("The human eye detects light.", "The eye perceives light through the retina.", "factual"),
          ("The Nile River is the longest river in Africa.", "The Nile is Africa’s longest river.", "factual"),
          ("The Atlantic Ocean separates America and Europe.", "The Atlantic Ocean lies between the Americas and Europe.", "factual"),
          ("The lungs help in breathing.", "The lungs facilitate gas exchange in respiration.", "factual"),
          ("Jupiter is the largest planet in the Solar System.", "Jupiter is the biggest planet in our solar system.", "factual"),
          ("Earth rotates once every 24 hours.", "The Earth completes one full rotation every 24 hours.", "factual"),
          ("The Milky Way is our galaxy.", "Earth is part of the Milky Way galaxy.", "factual"),
          ("Chlorophyll gives plants their green color.", "Plants appear green because of chlorophyll pigments.", "factual"),
          ("A triangle has three sides.", "A triangle is a polygon with three sides.", "factual"),
          ("Gold is a metal.", "Gold is a metallic element.", "factual"),
          ("Bananas grow on plants.", "Bananas are produced by large herbaceous plants.", "factual"),

          # -------- HALLUCINATION --------
          ("The Moon orbits the Sun.", "The Moon orbits the Earth, not the Sun.", "hallucination"),
          ("The brain is located in the chest.", "The brain is located in the skull, not the chest.", "hallucination"),
          ("The Atlantic Ocean is the smallest ocean.", "The Pacific is the largest; the Arctic is the smallest.", "hallucination"),
          ("Water freezes at 50°C.", "Water freezes at 0°C, not 50°C.", "hallucination"),
          ("The Sun is a planet.", "The Sun is a star, not a planet.", "hallucination"),
          ("Plants produce carbon monoxide.", "Plants produce oxygen, not carbon monoxide.", "hallucination"),
          ("The capital of Japan is Beijing.", "Tokyo is the capital of Japan.", "hallucination"),
          ("Blood carries electricity.", "Blood carries oxygen and nutrients, not electricity.", "hallucination"),
          ("The Great Wall is in India.", "The Great Wall is in China, not India.", "hallucination"),
          ("The Earth revolves around the Moon.", "Earth revolves around the Sun, not the Moon.", "hallucination"),
          ("Dogs are reptiles.", "Dogs are mammals, not reptiles.", "hallucination"),
          ("Bats are birds.", "Bats are mammals capable of flight.", "hallucination"),
          ("Einstein invented the light bulb.", "Thomas Edison is credited with inventing the light bulb.", "hallucination"),
          ("A square has five sides.", "A square has four equal sides.", "hallucination"),
          ("The formula for water is CO₂.", "Water’s chemical formula is H₂O.", "hallucination"),
          ("Bees produce milk.", "Bees produce honey, not milk.", "hallucination"),
          ("Sound travels faster than light.", "Light travels much faster than sound.", "hallucination"),
          ("The Eiffel Tower is in London.", "The Eiffel Tower is located in Paris, not London.", "hallucination"),
          ("Mount Everest is the smallest mountain.", "Mount Everest is the tallest mountain.", "hallucination"),
          ("Humans have 300 bones as adults.", "Adults have 206 bones, not 300.", "hallucination"),
          ("The heart has two chambers.", "The heart has four chambers.", "hallucination"),
          ("The Amazon River is in Africa.", "The Amazon River is in South America.", "hallucination"),
          ("Electric current is measured in liters.", "Electric current is measured in amperes.", "hallucination"),
          ("Human body temperature is 10°C.", "Normal body temperature is around 37°C.", "hallucination"),
          ("The Statue of Liberty is in Rome.", "The Statue of Liberty is in New York, not Rome.", "hallucination"),
          ("Plants emit carbon dioxide at night only.", "Plants respire CO₂ but photosynthesize O₂ in light.", "hallucination"),
          ("Light travels slower than sound.", "Light travels much faster than sound.", "hallucination"),
          ("Penguins live in the Arctic.", "Penguins live in the Southern Hemisphere, not the Arctic.", "hallucination"),
          ("Venus is the fifth planet from the Sun.", "Venus is the second planet from the Sun.", "hallucination"),
          ("Shakespeare wrote The Lord of the Rings.", "J.R.R. Tolkien wrote The Lord of the Rings.", "hallucination"),
          ("Photosynthesis happens in animals.", "Photosynthesis occurs in plants, not animals.", "hallucination"),
          ("The human heart is in the knee.", "The heart is located in the chest.", "hallucination"),
          ("The boiling point of water is 0°C.", "Water boils at 100°C, not 0°C.", "hallucination"),
          ("Oxygen is toxic to humans.", "Oxygen is essential for human life.", "hallucination"),
          ("The Sahara Desert is in South America.", "The Sahara is in Africa.", "hallucination"),
          ("Mars is known as the Blue Planet.", "Earth is called the Blue Planet.", "hallucination"),
          ("DNA is found only in plants.", "DNA is present in all living organisms.", "hallucination"),
          ("Saturn has no rings.", "Saturn has a prominent ring system.", "hallucination"),
          ("Humans see through their ears.", "Vision occurs through the eyes, not ears.", "hallucination"),
          ("The Nile River is in North America.", "The Nile is in Africa.", "hallucination"),
          ("The Pacific Ocean lies between Africa and Europe.", "The Atlantic lies between Africa and Europe.", "hallucination"),
          ("Humans breathe nitrogen only.", "Humans breathe oxygen; nitrogen is inert.", "hallucination"),
          ("Jupiter is the smallest planet.", "Jupiter is the largest planet.", "hallucination"),
          ("Earth rotates once per minute.", "Earth rotates once every 24 hours.", "hallucination"),
          ("The Milky Way is a solar system.", "The Milky Way is a galaxy, not a solar system.", "hallucination"),
          ("Chlorophyll is red in color.", "Chlorophyll is green, not red.", "hallucination"),
          ("A triangle has four sides.", "A triangle has three sides.", "hallucination"),
          ("Gold is a gas.", "Gold is a solid metal.", "hallucination"),
          ("Bananas grow underground.", "Bananas grow on plants above ground.", "hallucination"),
          
          ("The Moon reflects sunlight.", "Moonlight is sunlight reflected off the Moon’s surface.", "factual"),
          ("Atoms are made of protons, neutrons, and electrons.", "An atom consists of protons, neutrons, and electrons.", "factual"),
          ("The heart is an organ in the circulatory system.", "The circulatory system includes the heart and blood vessels.", "factual"),
          ("Sound needs a medium to travel.", "Sound waves require a material medium like air or water.", "factual"),
          ("The speed of sound is slower than light.", "Sound travels much slower than light.", "factual"),
          ("Clouds form from condensed water vapor.", "Clouds are made when water vapor condenses into droplets.", "factual"),
          ("Lightning is a discharge of static electricity.", "Lightning occurs when electrical charge builds up in clouds.", "factual"),
          ("The brain controls voluntary movement.", "Voluntary motion is regulated by the motor cortex in the brain.", "factual"),
          ("The lungs exchange oxygen and carbon dioxide.", "Gas exchange in humans happens in the alveoli of the lungs.", "factual"),
          ("Iron is a metal element.", "Iron is classified as a metallic chemical element.", "factual"),
          ("Earth has one natural satellite.", "The Moon is Earth’s only natural satellite.", "factual"),
          ("The human body contains water.", "The human body is composed of about 60% water.", "factual"),
          ("Carbon is found in all living things.", "All organic compounds contain carbon atoms.", "factual"),
          ("The ozone layer protects Earth from UV radiation.", "Ozone in the stratosphere absorbs harmful ultraviolet light.", "factual"),
          ("Seasons are caused by Earth's tilt.", "Earth’s axial tilt causes seasonal variations.", "factual"),
          ("Rainbows form due to light refraction and reflection.", "Rainbows appear when sunlight is refracted and reflected in raindrops.", "factual"),
          ("Electricity can flow through metals.", "Metals conduct electricity due to free electrons.", "factual"),
          ("Salt dissolves in water.", "Sodium chloride easily dissolves in water forming ions.", "factual"),
          ("Photosynthesis occurs in chloroplasts.", "Chloroplasts in plant cells carry out photosynthesis.", "factual"),
          ("Bacteria are single-celled organisms.", "Bacteria consist of a single cell with no nucleus.", "factual"),
          ("Earth's atmosphere contains oxygen and nitrogen.", "Air is mainly composed of nitrogen and oxygen.", "factual"),
          ("Earth’s core is mostly iron and nickel.", "The planet’s inner core is composed mainly of iron and nickel.", "factual"),
          ("Rain forms from condensed water droplets.", "Rain occurs when cloud droplets coalesce and fall.", "factual"),
          ("Energy cannot be created or destroyed.", "According to the law of conservation of energy, energy is constant.", "factual"),
          ("Blood pressure is measured in millimeters of mercury.", "The unit of blood pressure is mmHg.", "factual"),
          ("The kidneys filter waste from the blood.", "Kidneys remove waste and regulate fluid balance in the body.", "factual"),
          ("Earth’s gravity keeps the atmosphere in place.", "Gravity holds Earth’s gases close to the surface.", "factual"),
          ("Plants absorb water through their roots.", "Plant roots take in water and nutrients from the soil.", "factual"),
          ("Cells are the basic unit of life.", "Every living organism is made up of cells.", "factual"),
          ("Copper conducts electricity.", "Copper is widely used as an electrical conductor.", "factual"),
          ("Light travels in straight lines.", "In uniform media, light propagates in straight lines.", "factual"),
          ("Earth is the third planet from the Sun.", "Our planet occupies the third orbit from the Sun.", "factual"),
          ("Fish breathe through gills.", "Gills allow fish to extract oxygen from water.", "factual"),
          ("The metric system is used worldwide.", "Most countries use the metric system for measurement.", "factual"),
          ("The pancreas produces insulin.", "The pancreas secretes the hormone insulin to regulate sugar.", "factual"),
          ("Diamonds are made of carbon.", "Diamonds consist of carbon atoms arranged in a crystal lattice.", "factual"),
          ("Mitosis results in two identical cells.", "Cell division through mitosis creates identical daughter cells.", "factual"),
          ("Earth’s rotation causes day and night.", "The alternation of day and night is due to Earth’s rotation.", "factual"),
          ("Blood circulates continuously through the body.", "The circulatory system keeps blood moving throughout the body.", "factual"),
          ("Heat flows from hot to cold objects.", "Thermal energy transfers from warmer to cooler regions.", "factual"),
          ("Mars has two moons.", "Phobos and Deimos are the moons of Mars.", "factual"),
          ("Neptune is the farthest planet from the Sun.", "Neptune orbits the Sun at the greatest distance among planets.", "factual"),
          ("Mercury is the closest planet to the Sun.", "Mercury has the smallest orbit around the Sun.", "factual"),
          ("The Moon has no atmosphere.", "The Moon lacks a significant atmosphere.", "factual"),
          ("Venus has a thick carbon dioxide atmosphere.", "Venus’s dense atmosphere is composed mainly of CO₂.", "factual"),
          ("Jupiter is a gas giant.", "Jupiter is primarily made of hydrogen and helium gases.", "factual"),
          ("Rainforests have high biodiversity.", "Tropical rainforests are home to diverse species.", "factual"),
          ("Polar bears live in the Arctic.", "Polar bears inhabit the Arctic region.", "factual"),
          ("The equator divides Earth into two hemispheres.", "The equator separates the Northern and Southern Hemispheres.", "factual"),
          ("The human skeleton provides body support.", "Bones form the framework that supports the human body.", "factual"),
          ("Bees help in pollination.", "Bees transfer pollen between flowers aiding reproduction.", "factual"),
                      

          ("The Moon generates its own light.", "The Moon reflects sunlight; it doesn’t emit its own light.", "hallucination"),
          ("Atoms are made of light waves.", "Atoms consist of particles, not light waves.", "hallucination"),
          ("The heart is a bone.", "The heart is a muscular organ, not a bone.", "hallucination"),
          ("Sound can travel through a vacuum.", "Sound cannot travel in a vacuum because it needs a medium.", "hallucination"),
          ("Light travels slower than sound.", "Light travels much faster than sound.", "hallucination"),
          ("Clouds are made of smoke.", "Clouds form from condensed water vapor, not smoke.", "hallucination"),
          ("Lightning is caused by magnets.", "Lightning results from static electrical discharge, not magnets.", "hallucination"),
          ("The stomach controls voluntary movement.", "The brain controls voluntary motion, not the stomach.", "hallucination"),
          ("The lungs digest food.", "Digestion occurs in the stomach and intestines, not the lungs.", "hallucination"),
          ("Iron is a gas.", "Iron is a solid metal at room temperature.", "hallucination"),
          ("Earth has two moons.", "Earth has only one natural satellite, the Moon.", "hallucination"),
          ("The human body is made entirely of metal.", "The body consists mostly of water and organic compounds.", "hallucination"),
          ("Carbon is a liquid element.", "Carbon is a solid at normal temperatures.", "hallucination"),
          ("The ozone layer causes earthquakes.", "The ozone layer protects from UV radiation, not quakes.", "hallucination"),
          ("Seasons are caused by Earth’s distance from the Sun.", "Seasons are caused by the tilt, not distance.", "hallucination"),
          ("Rainbows form because of thunder.", "Rainbows form by light refraction, not thunder.", "hallucination"),
          ("Electricity cannot flow through metals.", "Metals are excellent conductors of electricity.", "hallucination"),
          ("Salt cannot dissolve in water.", "Salt easily dissolves in water.", "hallucination"),
          ("Photosynthesis happens in animals.", "Photosynthesis occurs only in plants and some bacteria.", "hallucination"),
          ("Bacteria are multicellular organisms.", "Bacteria are single-celled organisms.", "hallucination"),
          ("Earth’s atmosphere is made only of carbon dioxide.", "Earth’s air contains mostly nitrogen and oxygen.", "hallucination"),
          ("Earth’s core is made of ice.", "The core is primarily metal, not ice.", "hallucination"),
          ("Rain falls upward due to wind.", "Rain falls downward due to gravity.", "hallucination"),
          ("Energy can be destroyed.", "Energy is conserved and cannot be destroyed.", "hallucination"),
          ("Blood pressure is measured in kilometers.", "Blood pressure is measured in mmHg, not kilometers.", "hallucination"),
          ("The kidneys produce light.", "Kidneys filter blood, they don’t produce light.", "hallucination"),
          ("Earth has no gravity.", "Earth has gravity which holds everything to its surface.", "hallucination"),
          ("Plants absorb water through their leaves only.", "Most water absorption occurs through roots.", "hallucination"),
          ("Cells are smaller than atoms.", "Cells are much larger than atoms.", "hallucination"),
          ("Copper is an insulator.", "Copper is an electrical conductor, not an insulator.", "hallucination"),
          ("Light curves naturally in empty space.", "Light travels in straight lines in a vacuum.", "hallucination"),
          ("Earth is the fourth planet from the Sun.", "Earth is the third planet from the Sun.", "hallucination"),
          ("Fish breathe air directly through their nose.", "Fish breathe through gills, not noses.", "hallucination"),
          ("The imperial system is used worldwide.", "Most countries use the metric system.", "hallucination"),
          ("The pancreas pumps blood.", "The heart pumps blood, not the pancreas.", "hallucination"),
          ("Diamonds are made of ice.", "Diamonds are made of carbon.", "hallucination"),
          ("Mitosis produces different cells.", "Mitosis produces identical daughter cells.", "hallucination"),
          ("Night is caused by clouds blocking sunlight.", "Night occurs due to Earth’s rotation away from the Sun.", "hallucination"),
          ("Blood stays still inside the body.", "Blood circulates continuously through the body.", "hallucination"),
          ("Heat flows from cold to hot.", "Heat always flows from hot to cold.", "hallucination"),
          ("Mars has no moons.", "Mars has two moons, Phobos and Deimos.", "hallucination"),
          ("Neptune is the closest planet to the Sun.", "Mercury is the closest planet.", "hallucination"),
          ("Mercury is the farthest planet.", "Neptune is the farthest planet from the Sun.", "hallucination"),
          ("The Moon has a thick atmosphere.", "The Moon has no significant atmosphere.", "hallucination"),
          ("Venus has no atmosphere.", "Venus has a dense CO₂ atmosphere.", "hallucination"),
          ("Jupiter is made of solid rock.", "Jupiter is a gas giant.", "hallucination"),
          ("Rainforests have very few species.", "Rainforests are extremely biodiverse.", "hallucination"),
          ("Polar bears live in Antarctica.", "Polar bears live in the Arctic, not Antarctica.", "hallucination"),
          ("The equator is a mountain range.", "The equator is an imaginary line, not a mountain.", "hallucination"),
          ("The skeleton is made of plastic.", "The human skeleton is made of bone tissue.", "hallucination"),
          ("Bees eat rocks.", "Bees collect nectar and pollen, not rocks.", "hallucination"),
          
          ("The Sun rises in the east.", "The Sun appears to rise in the east due to Earth's rotation.", "factual"),
          ("A year has 12 months.", "There are 12 months in one calendar year.", "factual"),
          ("The freezing point of water is 0°C.", "Water freezes into ice at 0 degrees Celsius.", "factual"),
          ("Spiders have eight legs.", "All arachnids, including spiders, have eight legs.", "factual"),
          ("The Amazon rainforest is in South America.", "The Amazon rainforest spans several South American countries.", "factual"),
          ("Humans need oxygen to survive.", "Oxygen is essential for human respiration.", "factual"),
          ("Mercury is the smallest planet in the Solar System.", "Mercury is the smallest of all the planets.", "factual"),
          ("Electric current flows from positive to negative in circuits.", "Conventional current direction is from positive to negative.", "factual"),
          ("Humans have 32 adult teeth.", "An adult human typically has 32 permanent teeth.", "factual"),
          ("The Atlantic Ocean lies between America and Europe.", "The Atlantic Ocean separates the Americas from Europe and Africa.", "factual"),
          ("The Moon has craters on its surface.", "The Moon’s surface is covered with impact craters.", "factual"),
          ("A compass points toward the magnetic north.", "The needle of a compass aligns with Earth’s magnetic field.", "factual"),
          ("The Great Pyramid is in Egypt.", "The Great Pyramid of Giza is located in Egypt.", "factual"),
          ("Venus is the hottest planet in the Solar System.", "Due to its thick CO₂ atmosphere, Venus is the hottest planet.", "factual"),
          ("The lungs supply oxygen to the blood.", "Lungs oxygenate blood during respiration.", "factual"),
          ("A leap year occurs every four years.", "Leap years add one extra day every four years.", "factual"),
          ("Lightning is hotter than the surface of the Sun.", "A lightning bolt’s temperature exceeds that of the Sun’s surface.", "factual"),
          ("An octopus has eight arms.", "Octopuses possess eight limbs for movement and grasping.", "factual"),
          ("The Sahara Desert is in Africa.", "The Sahara is located in northern Africa.", "factual"),
          ("Paper is made from wood pulp.", "Most paper is produced using cellulose fibers from trees.", "factual"),
          ("The Earth rotates from west to east.", "Earth’s rotation direction is west to east.", "factual"),
          ("The Milky Way is a barred spiral galaxy.", "Our galaxy, the Milky Way, has a barred spiral structure.", "factual"),
          ("Albert Einstein was a physicist.", "Einstein was a theoretical physicist known for relativity.", "factual"),
          ("Water expands when it freezes.", "Unlike most substances, water expands upon freezing.", "factual"),
          ("Bees communicate through dance.", "Honeybees use the waggle dance to share food locations.", "factual"),
          ("The Pacific Ocean is deeper than the Atlantic Ocean.", "The Pacific Ocean contains the world’s deepest trenches.", "factual"),
          ("The Himalayas are the tallest mountain range.", "The Himalayas contain the highest peaks on Earth.", "factual"),
          ("The human brain is part of the nervous system.", "The brain is the central organ of the human nervous system.", "factual"),
          ("The Eiffel Tower was built in 1889.", "Paris’s Eiffel Tower was completed in 1889.", "factual"),
          ("Mount Kilimanjaro is in Africa.", "Mount Kilimanjaro is located in Tanzania, Africa.", "factual"),
          ("The Arctic Circle is near the North Pole.", "The Arctic Circle lies close to Earth’s northernmost region.", "factual"),
          ("The boiling point of water decreases at high altitudes.", "At higher altitudes, water boils below 100°C due to low pressure.", "factual"),
          ("Humans have DNA in their cells.", "Every human cell contains deoxyribonucleic acid.", "factual"),
          ("Oceans cover about 70% of Earth's surface.", "Most of Earth's surface area is covered by oceans.", "factual"),
          ("Blood contains red and white cells.", "Human blood is composed of red cells, white cells, and plasma.", "factual"),
          ("Penguins are birds that cannot fly.", "Penguins are flightless birds adapted for swimming.", "factual"),
          ("The human skeleton provides structure and support.", "Bones support the body and protect organs.", "factual"),
          ("The equator has a warm climate.", "Regions along the equator experience consistently warm temperatures.", "factual"),
          ("Cows are herbivores.", "Cows eat plants and grass, making them herbivores.", "factual"),
          ("The Sun is a source of light and heat for Earth.", "The Sun emits light and heat that sustain life on Earth.", "factual"),
          ("Mount Fuji is in Japan.", "Mount Fuji is Japan’s tallest mountain.", "factual"),
          ("Atoms are the building blocks of matter.", "All matter consists of atoms.", "factual"),
          ("Gold is a good conductor of electricity.", "Gold conducts electricity efficiently.", "factual"),
          ("Clouds are made of tiny water droplets.", "Clouds form from condensed water vapor.", "factual"),
          ("Jupiter has a giant red storm called the Great Red Spot.", "The Great Red Spot is a massive storm on Jupiter.", "factual"),
          ("Australia is both a country and a continent.", "Australia serves as both a country and a continent.", "factual"),
          ("Sharks have skeletons made of cartilage.", "Unlike fish, sharks have cartilage-based skeletons.", "factual"),
          ("Mars appears red due to iron oxide.", "The reddish color of Mars comes from surface iron oxide.", "factual"),
          ("The North Star is called Polaris.", "Polaris is the bright star near the celestial north pole.", "factual"),
          ("Earth has a magnetic field.", "Earth’s core generates a magnetic field that shields the planet.", "factual"),

          ("The Sun rises in the west.", "The Sun rises in the east, not the west.", "hallucination"),
          ("A year has 10 months.", "A calendar year has 12 months.", "hallucination"),
          ("Water freezes at 10°C.", "Water freezes at 0°C, not 10°C.", "hallucination"),
          ("Spiders have six legs.", "Spiders have eight legs, not six.", "hallucination"),
          ("The Amazon rainforest is in Asia.", "The Amazon rainforest is in South America.", "hallucination"),
          ("Humans can breathe underwater.", "Humans need oxygen from air, not water.", "hallucination"),
          ("Mercury is the largest planet.", "Jupiter is the largest planet.", "hallucination"),
          ("Electric current flows from negative to positive only.", "Conventional current flows from positive to negative.", "hallucination"),
          ("Humans have 40 adult teeth.", "Adults have 32 teeth, not 40.", "hallucination"),
          ("The Atlantic Ocean is larger than the Pacific Ocean.", "The Pacific Ocean is the largest.", "hallucination"),
          ("The Moon is made of cheese.", "The Moon is made of rock, not cheese.", "hallucination"),
          ("A compass points to the south.", "A compass points toward the magnetic north.", "hallucination"),
          ("The Great Pyramid is in Greece.", "The Great Pyramid is in Egypt.", "hallucination"),
          ("Venus is the coldest planet.", "Venus is the hottest planet.", "hallucination"),
          ("The lungs pump blood.", "The heart pumps blood, not the lungs.", "hallucination"),
          ("Leap years happen every 10 years.", "Leap years occur every 4 years.", "hallucination"),
          ("Lightning is colder than ice.", "Lightning is extremely hot, hotter than the Sun’s surface.", "hallucination"),
          ("An octopus has ten arms.", "An octopus has eight arms, not ten.", "hallucination"),
          ("The Sahara Desert is in Australia.", "The Sahara is located in Africa.", "hallucination"),
          ("Paper is made from metal.", "Paper is made from wood pulp, not metal.", "hallucination"),
          ("The Earth rotates from east to west.", "Earth rotates from west to east.", "hallucination"),
          ("The Milky Way is a single star.", "The Milky Way is a galaxy, not one star.", "hallucination"),
          ("Einstein was a famous chef.", "Einstein was a physicist, not a chef.", "hallucination"),
          ("Water contracts when it freezes.", "Water expands upon freezing.", "hallucination"),
          ("Bees communicate by sound waves only.", "Bees use dance and pheromones to communicate.", "hallucination"),
          ("The Atlantic Ocean is the deepest ocean.", "The Pacific Ocean is the deepest.", "hallucination"),
          ("The Himalayas are in Europe.", "The Himalayas are in Asia.", "hallucination"),
          ("The human brain is in the stomach.", "The brain is located in the skull.", "hallucination"),
          ("The Eiffel Tower is in Rome.", "The Eiffel Tower is in Paris.", "hallucination"),
          ("Mount Kilimanjaro is in Canada.", "Mount Kilimanjaro is in Tanzania, Africa.", "hallucination"),
          ("The Arctic Circle is near the equator.", "The Arctic Circle lies near the North Pole.", "hallucination"),
          ("Water boils faster on mountains.", "Water boils at lower temperatures at high altitudes.", "hallucination"),
          ("Humans have metal instead of DNA.", "Humans have DNA in their cells, not metal.", "hallucination"),
          ("Oceans cover only 10% of Earth’s surface.", "Oceans cover around 70% of Earth’s surface.", "hallucination"),
          ("Blood is blue inside the body.", "Blood is always red; oxygen changes its shade.", "hallucination"),
          ("Penguins can fly.", "Penguins are flightless birds.", "hallucination"),
          ("The skeleton is made of plastic.", "Bones are made of calcium, not plastic.", "hallucination"),
          ("The equator is cold all year.", "The equator has a warm climate.", "hallucination"),
          ("Cows eat meat.", "Cows are herbivores.", "hallucination"),
          ("The Sun produces no heat.", "The Sun emits both heat and light.", "hallucination"),
          ("Mount Fuji is in China.", "Mount Fuji is in Japan.", "hallucination"),
          ("Atoms are made of clouds.", "Atoms consist of protons, neutrons, and electrons.", "hallucination"),
          ("Gold is a non-conductor.", "Gold is an excellent conductor of electricity.", "hallucination"),
          ("Clouds are made of cotton.", "Clouds form from condensed water vapor.", "hallucination"),
          ("Jupiter has no storms.", "Jupiter’s Great Red Spot is a massive storm.", "hallucination"),
          ("Australia is part of Europe.", "Australia is a separate continent and country.", "hallucination"),
          ("Sharks have bones made of steel.", "Sharks have cartilage-based skeletons.", "hallucination"),
          ("Mars is green because of plants.", "Mars appears red due to iron oxide, not vegetation.", "hallucination"),
          ("The North Star is called Sirius.", "The North Star is Polaris, not Sirius.", "hallucination"),
          ("Earth has no magnetic field.", "Earth has a strong magnetic field.", "hallucination"),
          
          
# ---------- FACTUAL ----------
("Mercury is the closest planet to the Sun.", "Mercury orbits nearest to the Sun in our Solar System.", "factual"),
("The Moon reflects sunlight.", "The Moon shines because it reflects light from the Sun.", "factual"),
("The human heart beats to pump blood.", "Heartbeats move blood through the body.", "factual"),
("Carbon has atomic number 6.", "Carbon is the sixth element in the periodic table.", "factual"),
("The Amazon rainforest is in South America.", "The Amazon spans several South American countries.", "factual"),
("The human ear helps in hearing.", "Ears detect sound waves and help us hear.", "factual"),
("The brain processes sensory information.", "The human brain interprets signals from the senses.", "factual"),
("Water covers about 70% of Earth's surface.", "Nearly 70% of Earth is covered by oceans and water bodies.", "factual"),
("The currency of Japan is yen.", "Japan uses the yen as its official currency.", "factual"),
("The largest planet in our solar system is Jupiter.", "Jupiter is the biggest planet orbiting the Sun.", "factual"),
("The human liver detoxifies chemicals.", "The liver filters toxins from the bloodstream.", "factual"),
("Rainbows are formed by refraction of light.", "Light bends and disperses to form a rainbow.", "factual"),
("Sound requires a medium to travel.", "Sound waves need air, water, or solids to propagate.", "factual"),
("The Sahara Desert is located in Africa.", "The Sahara lies across northern Africa.", "factual"),
("The speed of sound is slower than light.", "Sound travels much slower than light.", "factual"),
("The periodic table organizes chemical elements.", "Elements are arranged by atomic number in the periodic table.", "factual"),
("The lungs absorb oxygen from the air.", "Lungs take in oxygen during respiration.", "factual"),
("The Eiffel Tower was built in the 19th century.", "The Eiffel Tower was constructed in the late 1800s.", "factual"),
("The heart is made of cardiac muscle.", "Cardiac muscle tissue makes up the human heart.", "factual"),
("The Great Barrier Reef is in Australia.", "Australia is home to the Great Barrier Reef.", "factual"),
("Venus has a dense carbon dioxide atmosphere.", "Venus’s atmosphere is mostly carbon dioxide.", "factual"),
("The Pacific Ocean borders Asia and the Americas.", "The Pacific lies between Asia and the Americas.", "factual"),
("Atoms consist of protons, neutrons, and electrons.", "An atom is made up of subatomic particles.", "factual"),
("Lightning is a form of electricity.", "Lightning is an electrical discharge in the atmosphere.", "factual"),
("Humans breathe oxygen and exhale carbon dioxide.", "Respiration involves inhaling oxygen and releasing CO₂.", "factual"),
("The brain controls the nervous system.", "The central nervous system is governed by the brain.", "factual"),
("The currency of the United Kingdom is the pound sterling.", "The British pound is the UK’s official currency.", "factual"),
("Bees help in pollination.", "Bees transfer pollen, aiding plant reproduction.", "factual"),
("Saturn has the most prominent ring system.", "Saturn’s visible rings are made of ice and rock.", "factual"),
("The pancreas produces insulin.", "Insulin is secreted by the pancreas.", "factual"),
("Oceans are saltwater bodies.", "Oceans contain saline water.", "factual"),
("Electricity flows through conductors.", "Conductors allow electric current to pass.", "factual"),
("The United States has 50 states.", "There are fifty states in the USA.", "factual"),
("Bacteria are single-celled organisms.", "Bacteria consist of a single cell.", "factual"),
("Ice is less dense than water.", "Ice floats because it is less dense than liquid water.", "factual"),
("The moon causes ocean tides.", "Tidal forces are caused by the Moon’s gravity.", "factual"),
("Humans need sleep to function properly.", "Sleep is essential for restoring energy and health.", "factual"),
("Volcanoes release lava during eruptions.", "Lava flows out of erupting volcanoes.", "factual"),
("A year has 12 months.", "There are twelve months in a calendar year.", "factual"),
("Copper is a good conductor of electricity.", "Electric current passes easily through copper.", "factual"),
("Sharks are fish.", "Sharks belong to the fish group, not mammals.", "factual"),
("Sound cannot travel in a vacuum.", "A vacuum lacks a medium for sound propagation.", "factual"),
("The human skin is the body’s largest organ.", "Skin is the largest organ in the human body.", "factual"),
("Clouds are made of condensed water vapor.", "Clouds consist of tiny droplets of water or ice.", "factual"),
("Caffeine is found in coffee.", "Coffee naturally contains caffeine.", "factual"),
("The Statue of Liberty was a gift from France.", "France gifted the Statue of Liberty to the USA.", "factual"),
("Penguins are flightless birds.", "Penguins cannot fly but are excellent swimmers.", "factual"),
("Honey is made by bees from nectar.", "Bees produce honey by processing flower nectar.", "factual"),
("Carbon dioxide is colorless and odorless.", "CO₂ has no color or smell.", "factual"),
("Pluto is classified as a dwarf planet.", "Pluto was reclassified as a dwarf planet in 2006.", "factual"),

# ---------- HALLUCINATION ----------
("Mercury is the farthest planet from the Sun.", "Mercury is the closest, not the farthest planet.", "hallucination"),
("The Moon generates its own light.", "The Moon reflects sunlight, it doesn’t produce it.", "hallucination"),
("The human heart is in the brain.", "The heart is in the chest, not the brain.", "hallucination"),
("Carbon has atomic number 12.", "Carbon’s atomic number is 6, not 12.", "hallucination"),
("The Amazon rainforest is in Africa.", "The Amazon is located in South America.", "hallucination"),
("The human ear helps in digestion.", "Ears aid in hearing, not digestion.", "hallucination"),
("The brain stores food.", "The brain processes information, not food.", "hallucination"),
("Water covers 30% of Earth’s surface.", "About 70% of Earth is covered by water.", "hallucination"),
("The currency of Japan is the won.", "Japan uses yen; South Korea uses won.", "hallucination"),
("The largest planet in our solar system is Mars.", "Jupiter, not Mars, is the largest planet.", "hallucination"),
("The human liver pumps blood.", "The heart pumps blood, not the liver.", "hallucination"),
("Rainbows are made of sound waves.", "Rainbows form from light, not sound.", "hallucination"),
("Sound travels faster than light.", "Light travels far faster than sound.", "hallucination"),
("The Sahara Desert is in South America.", "The Sahara is located in Africa.", "hallucination"),
("The speed of sound is faster in vacuum.", "Sound cannot travel in a vacuum.", "hallucination"),
("The periodic table lists animals.", "The periodic table lists chemical elements.", "hallucination"),
("The lungs store food.", "Lungs exchange gases, not store food.", "hallucination"),
("The Eiffel Tower is in Berlin.", "It is in Paris, not Berlin.", "hallucination"),
("The heart is a bone.", "The heart is a muscle, not a bone.", "hallucination"),
("The Great Barrier Reef is in India.", "It’s in Australia, not India.", "hallucination"),
("Venus has oxygen-rich air.", "Venus’s atmosphere is mostly carbon dioxide.", "hallucination"),
("The Pacific Ocean is the smallest ocean.", "It’s the largest, not the smallest.", "hallucination"),
("Atoms are made only of protons.", "Atoms also contain neutrons and electrons.", "hallucination"),
("Lightning is frozen water.", "Lightning is an electric discharge, not ice.", "hallucination"),
("Humans inhale carbon dioxide.", "Humans inhale oxygen, not carbon dioxide.", "hallucination"),
("The brain is part of the digestive system.", "It’s part of the nervous system.", "hallucination"),
("The currency of the UK is the euro.", "The UK uses the pound, not the euro.", "hallucination"),
("Bees feed on blood.", "Bees collect nectar, not blood.", "hallucination"),
("Saturn has no rings.", "Saturn’s ring system is highly visible.", "hallucination"),
("The pancreas pumps blood.", "The heart pumps blood; the pancreas produces insulin.", "hallucination"),
("Oceans contain fresh water.", "Oceans are saltwater bodies.", "hallucination"),
("Electricity flows through plastic.", "Plastic is an insulator, not a conductor.", "hallucination"),
("The USA has 49 states.", "There are 50 states in total.", "hallucination"),
("Bacteria are multicellular.", "Bacteria are single-celled.", "hallucination"),
("Ice sinks in water.", "Ice floats because it’s less dense.", "hallucination"),
("The moon has its own ocean tides.", "Tides occur on Earth, not the Moon.", "hallucination"),
("Humans can survive without sleep.", "Sleep is necessary for life.", "hallucination"),
("Volcanoes erupt snow.", "Volcanoes emit lava, not snow.", "hallucination"),
("A year has 10 months.", "There are 12 months in a year.", "hallucination"),
("Copper is an insulator.", "Copper is a conductor.", "hallucination"),
("Sharks are mammals.", "Sharks are fish, not mammals.", "hallucination"),
("Sound can travel in outer space.", "Sound cannot travel in a vacuum.", "hallucination"),
("The human skin is made of metal.", "Skin is biological tissue, not metal.", "hallucination"),
("Clouds are made of dust only.", "Clouds form from water vapor.", "hallucination"),
("Caffeine is a mineral.", "Caffeine is a stimulant compound, not a mineral.", "hallucination"),
("The Statue of Liberty is in Canada.", "It’s in New York, USA.", "hallucination"),
("Penguins can fly.", "Penguins are flightless birds.", "hallucination"),
("Honey is made from wood.", "Honey is made from flower nectar.", "hallucination"),
("Carbon dioxide has a strong smell.", "CO₂ is odorless.", "hallucination"),
("Pluto is the largest planet.", "Pluto is a dwarf planet, not the largest one.", "hallucination"),





        ]
    
    def _validate_pairs(self, pairs: List[Tuple]) -> List[Tuple]:
        """Validate pairs for quality and consistency"""
        validated = []
        for text, truth, label in pairs:
            if (self._validate_text_quality(text) and 
                self._validate_text_quality(truth)):
                validated.append((text, truth, label))
        return validated
    
    def _validate_text_quality(self, text: str) -> bool:
        """Validate text quality"""
        return len(text.strip()) > 10 and not text.isspace()
    
    def _balance_dataset(self, pairs: List[Tuple]) -> List[Tuple]:
        """Balance dataset"""
        factuals = [p for p in pairs if p[2] == "factual"]
        hallucinations = [p for p in pairs if p[2] == "hallucination"]
        
        min_count = min(len(factuals), len(hallucinations))
        if min_count == 0:
            return pairs
        
        balanced_factuals = random.sample(factuals, min_count)
        balanced_hallucinations = random.sample(hallucinations, min_count)
        
        balanced = balanced_factuals + balanced_hallucinations
        random.shuffle(balanced)
        return balanced

# ==============================================================
# Advanced Training with Proper System Integration
# ==============================================================

class AdaptiveContrastiveLoss:
    """Adaptive contrastive loss with temperature scaling"""
    
    def __init__(self, margin: float, temperature: float = 0.1):
        self.margin = margin
        self.temperature = temperature
    
    def __call__(self, similarities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute adaptive contrastive loss"""
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        scaled_sim = similarities / self.temperature
        
        pos_loss = -torch.log(torch.sigmoid(scaled_sim[pos_mask]) + 1e-8).mean() if pos_mask.any() else 0
        neg_loss = -torch.log(1 - torch.sigmoid(scaled_sim[neg_mask] - self.margin) + 1e-8).mean() if neg_mask.any() else 0
        
        return (pos_loss + neg_loss) / 2

class SmartTrainer:
    """Advanced trainer with proper system integration"""
    
    def __init__(self, config: LSDConfig, system_manager: SystemManager, logger: EnhancedLogger):
        self.config = config
        self.system_manager = system_manager
        self.device = system_manager.device
        self.directories = system_manager.directories
        self.logger = logger
        self.loss_fn = AdaptiveContrastiveLoss(margin=config.margin)
    
    def train(self, pairs: List[Tuple]) -> Tuple[Any, Any, nn.Module, nn.Module]:
        """Advanced training with comprehensive monitoring"""
        # Dataset preparation
        train_pairs, val_pairs = self._split_data(pairs)
        
        train_loader = DataLoader(
            PairDataset(train_pairs),
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            PairDataset(val_pairs),
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        self.logger.log(f"Training on {len(train_pairs)} samples, validating on {len(val_pairs)} samples", "INFO")
        
        # Model initialization
        extractor = EfficientHuggingFaceExtractor(
            self.config.model_name, self.device, self.config.max_length
        )
        truth_encoder = RobustTruthEncoder(self.config.truth_encoder_name, self.device)
        
        # Get dimensions safely
        with torch.no_grad():
            dummy_hidden = extractor.get_hidden_states(["dummy text"])
            dummy_truth = truth_encoder.encode_batch(["dummy truth"])
            
            hidden_dim = dummy_hidden.shape[-1]
            truth_dim = dummy_truth.shape[-1]
        
        self.logger.log(f"Model dimensions - Hidden: {hidden_dim}, Truth: {truth_dim}", "INFO")
        
        # Initialize projection heads
        hidden_proj = AdvancedProjectionHead(hidden_dim, self.config.shared_dim)
        truth_proj = AdvancedProjectionHead(truth_dim, self.config.shared_dim)
        
        hidden_proj.to(self.device)
        truth_proj.to(self.device)
        
        # Optimizer only for projection heads
        params = list(hidden_proj.parameters()) + list(truth_proj.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config.epochs
        )
        
        # Training loop
        best_val_loss = float('inf')
        history = []
        
        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._train_epoch(
                epoch, train_loader, extractor, truth_encoder, 
                hidden_proj, truth_proj, optimizer
            )
            
            val_metrics = self._validate_epoch(
                val_loader, extractor, truth_encoder, hidden_proj, truth_proj
            )
            
            scheduler.step()
            
            current_lr = scheduler.get_last_lr()[0]
            history.append({**train_metrics, **val_metrics, 'epoch': epoch, 'learning_rate': current_lr})
            
            self.logger.log(
                f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}, "
                f"val_loss={val_metrics['val_loss']:.4f}, "
                f"val_acc={val_metrics['val_accuracy']:.3f}, "
                f"lr={current_lr:.2e}",
                "INFO"
            )
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self._save_checkpoint(hidden_proj, truth_proj, epoch, val_metrics)
                self.logger.log(f"Saved best model with val_loss={val_metrics['val_loss']:.4f}", "SUCCESS")
        
        # Load best model
        self._load_best_checkpoint(hidden_proj, truth_proj)
        
        # Save training history
        self._save_training_history(history)
        
        return extractor, truth_encoder, hidden_proj, truth_proj
    
    def _train_epoch(self, epoch: int, loader: DataLoader, extractor: Any, 
                    truth_encoder: Any, hidden_proj: nn.Module, 
                    truth_proj: nn.Module, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Train for one epoch"""
        hidden_proj.train()
        truth_proj.train()
        
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch} Training", leave=False)
        
        for texts, truths, labels, _ in pbar:
            labels = labels.to(self.device)
            
            # Extract features with no gradients
            with torch.no_grad():
                hidden_states = extractor.get_hidden_states(texts)
                last_hidden = hidden_states[:, -1, :]
                truth_embs = truth_encoder.encode_batch(truths)
            
            # Project to shared space - gradients flow here
            hidden_projected = hidden_proj(last_hidden)
            truth_projected = truth_proj(truth_embs)
            
            # Compute similarities and loss
            similarities = F.cosine_similarity(hidden_projected, truth_projected, dim=-1)
            loss = self.loss_fn(similarities, labels)
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(hidden_proj.parameters()) + list(truth_proj.parameters()),
                max_norm=self.config.gradient_clip
            )
            optimizer.step()
            
            # Metrics
            predictions = (similarities > 0).float()
            accuracy = (predictions == labels).float().mean()
            
            total_loss += loss.item()
            total_acc += accuracy.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy.item():.3f}'
            })
        
        return {
            'train_loss': total_loss / num_batches,
            'train_accuracy': total_acc / num_batches
        }
    
    def _validate_epoch(self, loader: DataLoader, extractor: Any,
                       truth_encoder: Any, hidden_proj: nn.Module,
                       truth_proj: nn.Module) -> Dict[str, float]:
        """Validate for one epoch"""
        hidden_proj.eval()
        truth_proj.eval()
        
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        with torch.no_grad():
            for texts, truths, labels, _ in loader:
                labels = labels.to(self.device)
                
                hidden_states = extractor.get_hidden_states(texts)
                last_hidden = hidden_states[:, -1, :]
                truth_embs = truth_encoder.encode_batch(truths)
                
                hidden_projected = hidden_proj(last_hidden)
                truth_projected = truth_proj(truth_embs)
                
                similarities = F.cosine_similarity(hidden_projected, truth_projected, dim=-1)
                loss = self.loss_fn(similarities, labels)
                
                predictions = (similarities > 0).float()
                accuracy = (predictions == labels).float().mean()
                
                total_loss += loss.item()
                total_acc += accuracy.item()
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_accuracy': total_acc / num_batches
        }
    
    def _save_checkpoint(self, hidden_proj: nn.Module, truth_proj: nn.Module,
                        epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'hidden_proj_state_dict': hidden_proj.state_dict(),
            'truth_proj_state_dict': truth_proj.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, self.directories['models'] / "best_model.pt")
    
    def _load_best_checkpoint(self, hidden_proj: nn.Module, truth_proj: nn.Module):
        """Load best model checkpoint"""
        try:
            checkpoint_path = self.directories['models'] / "best_model.pt"
            checkpoint = torch.load(checkpoint_path)
            hidden_proj.load_state_dict(checkpoint['hidden_proj_state_dict'])
            truth_proj.load_state_dict(checkpoint['truth_proj_state_dict'])
            self.logger.log(f"Loaded best model from epoch {checkpoint['epoch']}", "SUCCESS")
        except FileNotFoundError:
            self.logger.log("No checkpoint found, using final model", "WARNING")
    
    def _save_training_history(self, history: List[Dict]):
        """Save training history"""
        df = pd.DataFrame(history)
        df.to_csv(self.directories['results'] / "training_history.csv", index=False)
        self.logger.log("Saved training history", "SUCCESS")
    
    def _split_data(self, pairs: List[Tuple], train_ratio: float = 0.8) -> Tuple[List, List]:
        """Split data with stratification"""
        factuals = [p for p in pairs if p[2] == "factual"]
        hallucinations = [p for p in pairs if p[2] == "hallucination"]
        
        train_size = int(len(factuals) * train_ratio)
        train_factuals = factuals[:train_size]
        val_factuals = factuals[train_size:]
        
        train_hallucinations = hallucinations[:train_size]
        val_hallucinations = hallucinations[train_size:]
        
        train_pairs = train_factuals + train_hallucinations
        val_pairs = val_factuals + val_hallucinations
        
        random.shuffle(train_pairs)
        random.shuffle(val_pairs)
        
        return train_pairs, val_pairs

# ==============================================================
# Supporting Classes
# ==============================================================

class PairDataset(Dataset):
    def __init__(self, pairs): 
        self.pairs = pairs
        
    def __len__(self): 
        return len(self.pairs)
        
    def __getitem__(self, i):
        text, truth, label_str = self.pairs[i]
        return {
            "text": text, 
            "truth": truth, 
            "label": 1 if label_str == "factual" else 0, 
            "label_str": label_str
        }

def collate_fn(batch):
    texts = [x["text"] for x in batch]
    truths = [x["truth"] for x in batch]
    labels = torch.tensor([x["label"] for x in batch])
    label_strs = [x["label_str"] for x in batch]
    return texts, truths, labels, label_strs

# ==============================================================
# Main Execution
# ==============================================================

def main():
    """Main execution with robust error handling"""
    # Initialize system
    config = LSDConfig()
    system_manager = SystemManager(config)
    logger = EnhancedLogger(system_manager.directories['logs'] / "analysis.log")
    
    try:
        with logger.task_context("Complete LSD Analysis"):
            
            # Step 1: Build quality dataset
            with logger.task_context("Dataset Construction"):
                dataset_builder = QualityDatasetBuilder(config)
                pairs = dataset_builder.build_quality_pairs()
                logger.log(f"Built dataset with {len(pairs)} quality pairs", "SUCCESS")
                logger.log(f"Factual: {len([p for p in pairs if p[2]=='factual'])}, "
                          f"Hallucination: {len([p for p in pairs if p[2]=='hallucination'])}", "INFO")
            
            # Step 2: Train models
            with logger.task_context("Model Training"):
                trainer = SmartTrainer(config, system_manager, logger)
                extractor, truth_encoder, hidden_proj, truth_proj = trainer.train(pairs)
                logger.log("Model training completed successfully", "SUCCESS")
            
            logger.log("All tasks completed successfully!", "SUCCESS")
            
            # Print final summary
            print("\n" + "="*60)
            print("LAYER-WISE SEMANTIC DYNAMICS ANALYSIS COMPLETE")
            print("="*60)
            print(f"Results saved to: {system_manager.directories['base']}")
            print(f"Training history: {system_manager.directories['results']}/training_history.csv")
            print(f"Best model: {system_manager.directories['models']}/best_model.pt")
            
    except Exception as e:
        logger.log(f"Analysis failed: {str(e)}", "ERROR")
        import traceback
        logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")

if __name__ == "__main__":
    main()