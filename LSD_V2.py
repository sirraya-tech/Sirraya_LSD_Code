# ==============================================================
# Layer-wise Semantic Dynamics (LSD)
# Complete Fixed Implementation
# ==============================================================
# !pip install -q transformers sentence-transformers datasets torch tqdm matplotlib scikit-learn scipy seaborn

import os, math, random
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from tqdm.auto import tqdm
from scipy.stats import ttest_ind, pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import datasets
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

# ==============================================================
# Enhanced Configuration
# ==============================================================
@dataclass
class LSDConfig:
    """Configuration for Layer-wise Semantic Dynamics analysis"""
    model_name: str = "gpt2"
    truth_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    shared_dim: int = 256
    batch_size: int = 4
    epochs: int = 20
    learning_rate: float = 1e-4
    margin: float = 0.3
    max_length: int = 128
    datasets: List[str] = None
    num_pairs: int = 200
    seed: int = 42
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ["synthetic", "truthfulqa"]  # Use multiple data sources

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
config = LSDConfig()

SAVE_DIR = Path("lsd_trained_enhanced"); SAVE_DIR.mkdir(exist_ok=True)
PLOT_DIR = SAVE_DIR / "plots"; PLOT_DIR.mkdir(exist_ok=True)
RESULTS_DIR = SAVE_DIR / "results"; RESULTS_DIR.mkdir(exist_ok=True)

# ==============================================================
# Utilities
# ==============================================================
def mean_pool_hidden(hidden, attn_mask):
    """Mean pool hidden states using attention mask."""
    mask = attn_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

class Logger:
    """Enhanced logging utility"""
    def __init__(self, log_file: Path = None):
        self.log_file = log_file or SAVE_DIR / "training.log"
        
    def log(self, message: str, print_message: bool = True):
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")
        
        if print_message:
            print(log_entry)

logger = Logger()

# ==============================================================
# Extractors
# ==============================================================
class HuggingFaceExtractor:
    """Extract layer-wise hidden states from a HuggingFace model."""
    def __init__(self, model_name=config.model_name, device=DEVICE, max_length=config.max_length):
        self.device = device
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
        logger.log(f"Initialized HuggingFaceExtractor with {self.num_layers} layers, hidden_size={self.hidden_size}")

    def get_hidden_states(self, texts):
        """Get hidden states for all layers with proper masking"""
        toks = self.tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length
        ).to(self.device)
        with torch.no_grad():
            outs = self.model(**toks)
        # Stack all hidden states and mean pool
        hidden_states = []
        for layer_hidden in outs.hidden_states:
            pooled = mean_pool_hidden(layer_hidden, toks["attention_mask"])
            hidden_states.append(pooled)
        return torch.stack(hidden_states, dim=1)  # [batch, layers, hidden_dim]

class TruthEncoder:
    """Encodes factual truth sentences."""
    def __init__(self, name=config.truth_encoder_name, device=DEVICE):
        logger.log(f"Loading truth encoder: {name}")
        self.model = SentenceTransformer(name).to(device)
        logger.log(f"Initialized TruthEncoder: {name}")
        
    def encode_batch(self, texts):
        """Encode texts with normalization"""
        emb = self.model.encode(texts, convert_to_tensor=True).to(DEVICE)
        return F.normalize(emb, p=2, dim=-1)

# ==============================================================
# Projection Heads
# ==============================================================
def build_proj(d1, d2):
    """Projection heads to align model and truth embeddings."""
    h = nn.Sequential(
        nn.Linear(d1, config.shared_dim*2), 
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(config.shared_dim*2, config.shared_dim), 
        nn.LayerNorm(config.shared_dim)
    ).to(DEVICE)
    
    t = nn.Sequential(
        nn.Linear(d2, config.shared_dim*2), 
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(config.shared_dim*2, config.shared_dim), 
        nn.LayerNorm(config.shared_dim)
    ).to(DEVICE)
    
    logger.log(f"Built projection heads: hidden_dim={d1}->{config.shared_dim}, truth_dim={d2}->{config.shared_dim}")
    return h, t

# ==============================================================
# Enhanced Dataset Builder with Multiple Sources
# ==============================================================
def build_pairs(max_per=config.num_pairs):
    """Build dataset pairs from multiple sources including TruthfulQA"""
    pairs = []
    dataset_stats = {}
    
    # Comprehensive synthetic dataset
    synthetic_pairs = [
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

            # -------- FACTUAL --------
        ("The Earth has one moon.", "Earth has a single natural satellite called the Moon.", "factual"),
        ("Mercury is the closest planet to the Sun.", "Mercury is the innermost planet in the Solar System.", "factual"),
        ("The ocean contains salt water.", "Seawater is saline in nature.", "factual"),
        ("A rainbow shows seven colors.", "A rainbow typically displays seven visible colors.", "factual"),
        ("Humans need water to survive.", "Water is essential for human life.", "factual"),
        ("The capital of Italy is Rome.", "Rome is the capital city of Italy.", "factual"),
        ("The human nose detects smell.", "The olfactory system enables humans to smell.", "factual"),
        ("The sky appears blue due to scattering.", "Rayleigh scattering causes the blue color of the sky.", "factual"),
        ("Earth’s atmosphere contains oxygen.", "Oxygen makes up about 21% of Earth’s atmosphere.", "factual"),
        ("Fish breathe through gills.", "Gills allow fish to extract oxygen from water.", "factual"),
        ("The Arctic is located at the North Pole.", "The Arctic region surrounds the North Pole.", "factual"),
        ("The human tongue helps in tasting food.", "Taste buds on the tongue detect flavors.", "factual"),
        ("The Pyramids of Giza are in Egypt.", "The Great Pyramids are located near Cairo, Egypt.", "factual"),
        ("The liver helps detoxify the body.", "The liver filters toxins from the bloodstream.", "factual"),
        ("The ozone layer protects from UV rays.", "Ozone in the atmosphere absorbs harmful ultraviolet radiation.", "factual"),
        ("Whales are mammals.", "Whales are warm-blooded marine mammals.", "factual"),
        ("Venus is hotter than Earth.", "Venus has a dense CO₂ atmosphere that traps heat.", "factual"),
        ("A compass points north.", "A magnetic compass aligns toward Earth’s magnetic north.", "factual"),
        ("Humans have five fingers on each hand.", "Each human hand typically has five fingers.", "factual"),
        ("The equator divides the Earth in half.", "The equator separates the Northern and Southern Hemispheres.", "factual"),
        ("The pancreas produces insulin.", "The pancreas secretes the hormone insulin.", "factual"),
        ("Antarctica is the coldest continent.", "Antarctica has the lowest average temperatures on Earth.", "factual"),
        ("Birds lay eggs.", "Birds reproduce by laying eggs.", "factual"),
        ("The Amazon rainforest is in South America.", "The Amazon rainforest spans several South American countries.", "factual"),
        ("Clouds form from water vapor.", "Condensed water vapor forms clouds.", "factual"),
        ("The brain uses electrical signals.", "Neurons communicate via electrical impulses.", "factual"),
        ("The moon affects ocean tides.", "Tidal forces are influenced by the Moon’s gravity.", "factual"),
        ("Humans have two lungs.", "The human body has two lungs for respiration.", "factual"),
        ("The Pacific Ocean touches Asia and America.", "The Pacific borders Asia and the Americas.", "factual"),
        ("Lightning is an electrical discharge.", "Lightning is a sudden electrostatic discharge.", "factual"),
        ("The skin is the largest organ.", "Human skin is the body’s largest organ.", "factual"),
        ("The Earth’s core is very hot.", "Earth’s inner core reaches extremely high temperatures.", "factual"),
        ("Atoms make up all matter.", "Everything is composed of atoms.", "factual"),
        ("The color of chlorophyll is green.", "Chlorophyll reflects green light.", "factual"),
        ("Sharks are fish.", "Sharks are cartilaginous fish.", "factual"),
        ("The Moon has craters.", "Impacts on the Moon’s surface form craters.", "factual"),
        ("The human ear helps with balance.", "The inner ear contains organs for hearing and balance.", "factual"),
        ("Steel is stronger than wood.", "Steel has higher tensile strength than wood.", "factual"),
        ("The smallest planet is Mercury.", "Mercury is the smallest planet in the Solar System.", "factual"),
        ("Fire needs oxygen to burn.", "Combustion requires oxygen.", "factual"),
        ("The Internet connects computers globally.", "The Internet links computers through a global network.", "factual"),
        ("The Taj Mahal is in India.", "The Taj Mahal is located in Agra, India.", "factual"),
        ("Honeybees live in colonies.", "Honeybees form organized colonies.", "factual"),
        ("Glass is made from sand.", "Glass is produced by melting silica sand.", "factual"),
        ("The thermometer measures temperature.", "A thermometer is used to measure heat or cold.", "factual"),
        ("The Earth is round.", "Earth is an oblate spheroid in shape.", "factual"),
        ("Diamonds are made of carbon.", "Diamonds consist of crystalline carbon.", "factual"),
        ("Volcanoes erupt molten rock.", "Volcanoes expel lava and ash during eruptions.", "factual"),
        ("Spiders have eight legs.", "Spiders are arachnids with eight legs.", "factual"),
        ("Humans have two eyes.", "People typically have two eyes for binocular vision.", "factual"),


        # -------- HALLUCINATION --------
        ("The Earth has two moons.", "Earth has only one natural satellite.", "hallucination"),
        ("Mercury is the coldest planet.", "Mercury is hot due to proximity to the Sun; Neptune is coldest.", "hallucination"),
        ("The ocean is made of fresh water.", "The ocean contains salt water.", "hallucination"),
        ("A rainbow has five colors.", "A rainbow typically shows seven colors.", "hallucination"),
        ("Humans can live without water.", "Humans cannot survive long without water.", "hallucination"),
        ("The capital of Italy is Venice.", "Rome is the capital of Italy.", "hallucination"),
        ("The nose is used for hearing.", "The nose detects smell, not sound.", "hallucination"),
        ("The sky is blue because of oceans.", "The sky is blue due to light scattering, not ocean reflection.", "hallucination"),
        ("Earth’s atmosphere is mostly carbon dioxide.", "Earth’s air is mostly nitrogen and oxygen.", "hallucination"),
        ("Fish breathe air with lungs.", "Fish use gills to extract oxygen from water.", "hallucination"),
        ("The Arctic is at the South Pole.", "The Arctic is at the North Pole.", "hallucination"),
        ("The tongue helps you see colors.", "The tongue detects taste, not sight.", "hallucination"),
        ("The Pyramids are in Mexico.", "The Egyptian pyramids are in Egypt, not Mexico.", "hallucination"),
        ("The liver pumps blood.", "The heart pumps blood, not the liver.", "hallucination"),
        ("The ozone layer causes UV rays.", "The ozone layer blocks UV rays.", "hallucination"),
        ("Whales are fish.", "Whales are mammals, not fish.", "hallucination"),
        ("Venus is colder than Neptune.", "Neptune is colder; Venus is extremely hot.", "hallucination"),
        ("A compass points south.", "A compass points north.", "hallucination"),
        ("Humans have six fingers normally.", "Humans typically have five fingers.", "hallucination"),
        ("The equator runs through the poles.", "The equator is halfway between the poles.", "hallucination"),
        ("The pancreas digests sound.", "The pancreas produces enzymes and insulin.", "hallucination"),
        ("Antarctica is the hottest continent.", "Antarctica is the coldest.", "hallucination"),
        ("Birds give birth to live young.", "Birds lay eggs.", "hallucination"),
        ("The Amazon rainforest is in Africa.", "It is in South America.", "hallucination"),
        ("Clouds are made of smoke.", "Clouds consist of condensed water vapor.", "hallucination"),
        ("The brain produces oxygen.", "The lungs handle oxygen exchange.", "hallucination"),
        ("The moon has its own light source.", "The Moon reflects sunlight.", "hallucination"),
        ("Humans have three lungs.", "Humans have two lungs.", "hallucination"),
        ("The Pacific Ocean is between Africa and Europe.", "The Atlantic separates Africa and Europe.", "hallucination"),
        ("Lightning is frozen air.", "Lightning is an electrical discharge.", "hallucination"),
        ("The skin is an internal organ.", "The skin is external.", "hallucination"),
        ("The Earth’s core is made of ice.", "Earth’s core is molten metal.", "hallucination"),
        ("Atoms are visible to the naked eye.", "Atoms are microscopic.", "hallucination"),
        ("Chlorophyll is purple.", "Chlorophyll is green.", "hallucination"),
        ("Sharks are mammals.", "Sharks are fish.", "hallucination"),
        ("The Moon has an atmosphere like Earth.", "The Moon has no significant atmosphere.", "hallucination"),
        ("The ear is used for taste.", "The ear detects sound and balance, not taste.", "hallucination"),
        ("Wood is stronger than steel.", "Steel is stronger than wood.", "hallucination"),
        ("Pluto is the largest planet.", "Jupiter is the largest planet.", "hallucination"),
        ("Fire burns without oxygen.", "Fire needs oxygen to burn.", "hallucination"),
        ("The Internet is a single computer.", "The Internet is a global network.", "hallucination"),
        ("The Taj Mahal is in Pakistan.", "The Taj Mahal is in India.", "hallucination"),
        ("Bees produce plastic.", "Bees produce honey, not plastic.", "hallucination"),
        ("Glass is a liquid metal.", "Glass is an amorphous solid made from sand.", "hallucination"),
        ("A thermometer measures distance.", "A thermometer measures temperature.", "hallucination"),
        ("The Earth is flat.", "Earth is round (an oblate sphere).", "hallucination"),
        ("Diamonds are made of water.", "Diamonds are made of carbon.", "hallucination"),
        ("Volcanoes erupt ice.", "Volcanoes erupt molten rock.", "hallucination"),
        ("Spiders have six legs.", "Spiders have eight legs.", "hallucination"),
        ("Humans have four eyes.", "Humans have two eyes.", "hallucination"),
        
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



    ]
    
    pairs.extend(synthetic_pairs)
    dataset_stats["synthetic"] = len(synthetic_pairs)
    
    # Load TruthfulQA dataset
    if "truthfulqa" in config.datasets:
        try:
            logger.log("Loading TruthfulQA dataset...")
            truthfulqa = datasets.load_dataset("domenicrosati/TruthfulQA", split="validation")
            truthfulqa_pairs = []
            
            for example in truthfulqa:
                question = example.get("Question", "").strip()
                correct_answers = example.get("Correct Answers", [])
                incorrect_answers = example.get("Incorrect Answers", [])
                
                # Add factual pairs from correct answers
                for correct_answer in correct_answers[:2]:  # Use up to 2 correct answers
                    if question and correct_answer:
                        truthfulqa_pairs.append((question, correct_answer, "factual"))
                
                # Add hallucination pairs from incorrect answers
                for incorrect_answer in incorrect_answers[:2]:  # Use up to 2 incorrect answers
                    if question and incorrect_answer:
                        truthfulqa_pairs.append((question, incorrect_answer, "hallucination"))
            
            if truthfulqa_pairs:
                pairs.extend(truthfulqa_pairs)
                dataset_stats["truthfulqa"] = len(truthfulqa_pairs)
                logger.log(f"Loaded TruthfulQA: {len(truthfulqa_pairs)} pairs")
            else:
                logger.log("TruthfulQA dataset loaded but no valid pairs found")
                
        except Exception as e:
            logger.log(f"TruthfulQA load failed: {e}")
    
    # Balance classes
    factuals = [p for p in pairs if p[2] == "factual"]
    hallucinations = [p for p in pairs if p[2] == "hallucination"]
    
    logger.log(f"Before balancing: {len(factuals)} factual, {len(hallucinations)} hallucination")
    
    min_count = min(len(factuals), len(hallucinations))
    if min_count > 0:
        factuals = factuals[:min_count]
        hallucinations = hallucinations[:min_count]
        balanced_pairs = factuals + hallucinations
        random.shuffle(balanced_pairs)
        final_pairs = balanced_pairs[:max_per]
    else:
        final_pairs = pairs[:max_per]
    
    logger.log(f"Final dataset: {len(final_pairs)} pairs")
    logger.log(f"Dataset statistics: {dataset_stats}")
    
    return final_pairs

# ==============================================================
# Training Components
# ==============================================================
def contrastive_loss(cos_sim, labels, margin=config.margin):
    """Margin-based contrastive loss."""
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    pos_loss = (1 - cos_sim[pos_mask]).mean() if pos_mask.any() else 0
    neg_loss = F.relu(cos_sim[neg_mask] - (-margin)).mean() if neg_mask.any() else 0
    
    return 0.5 * (pos_loss + neg_loss)

class PairDataset(Dataset):
    """Dataset for text-truth pairs"""
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
    """Collate function for DataLoader"""
    texts = [x["text"] for x in batch]
    truths = [x["truth"] for x in batch]
    labels = torch.tensor([x["label"] for x in batch]).to(DEVICE)
    label_strs = [x["label_str"] for x in batch]
    return texts, truths, labels, label_strs

# ==============================================================
# Enhanced Training (FIXED)
# ==============================================================
def train(pairs):
    """Train projection heads using factual vs hallucinated pairs"""
    if len(pairs) < config.batch_size * 2:
        logger.log(f"Warning: Very small dataset ({len(pairs)} pairs). Consider increasing dataset size.")
    
    # Split data
    n = int(0.8 * len(pairs))  # 80-20 split
    tr_pairs, va_pairs = pairs[:n], pairs[n:]
    
    train_dataset = PairDataset(tr_pairs)
    val_dataset = PairDataset(va_pairs)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    logger.log(f"Training on {len(tr_pairs)} samples, validating on {len(va_pairs)} samples")

    # Initialize models
    logger.log("Initializing models...")
    ex = HuggingFaceExtractor()
    te = TruthEncoder()
    
    # Get dimensions safely
    with torch.no_grad():
        dummy_text = ["test sentence"]
        dummy_hidden = ex.get_hidden_states(dummy_text)
        d1 = dummy_hidden.shape[-1]  # hidden dimension
        
        dummy_truth = te.encode_batch(dummy_text)
        d2 = dummy_truth.shape[-1]  # truth embedding dimension
    
    logger.log(f"Model dimensions - Hidden: {d1}, Truth: {d2}")
    
    # Build projection heads
    h_proj, t_proj = build_proj(d1, d2)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(h_proj.parameters()) + list(t_proj.parameters()), 
        lr=config.learning_rate, 
        weight_decay=1e-5
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # Training history
    history = []
    best_val_loss = float('inf')
    
    logger.log("Starting training...")
    
    for epoch in range(1, config.epochs + 1):
        # Training phase
        h_proj.train()
        t_proj.train()
        train_losses = []
        
        for texts, truths, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", leave=False):
            # Get model hidden states (last layer only for training)
            with torch.no_grad():
                hidden_states = ex.get_hidden_states(texts)  # [batch, layers, hidden_dim]
                last_layer_hidden = hidden_states[:, -1, :]  # [batch, hidden_dim]
                truth_embeddings = te.encode_batch(truths)   # [batch, truth_dim]
            
            # Project to shared space
            Hp = F.normalize(h_proj(last_layer_hidden), p=2, dim=-1)  # [batch, shared_dim]
            Gp = F.normalize(t_proj(truth_embeddings), p=2, dim=-1)   # [batch, shared_dim]
            
            # Compute cosine similarities and loss
            cos_sim = F.cosine_similarity(Hp, Gp, dim=-1)  # [batch]
            loss = contrastive_loss(cos_sim, labels)
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(h_proj.parameters()) + list(t_proj.parameters()), 
                max_norm=1.0
            )
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation phase
        h_proj.eval()
        t_proj.eval()
        val_losses = []
        val_accuracies = []
        
        with torch.no_grad():
            for texts, truths, labels, _ in val_loader:
                hidden_states = ex.get_hidden_states(texts)
                last_layer_hidden = hidden_states[:, -1, :]
                truth_embeddings = te.encode_batch(truths)
                
                Hp = F.normalize(h_proj(last_layer_hidden), p=2, dim=-1)
                Gp = F.normalize(t_proj(truth_embeddings), p=2, dim=-1)
                
                cos_sim = F.cosine_similarity(Hp, Gp, dim=-1)
                loss = contrastive_loss(cos_sim, labels)
                val_losses.append(loss.item())
                
                # Calculate accuracy
                preds = (cos_sim > 0).float()
                accuracy = (preds == labels).float().mean().item()
                val_accuracies.append(accuracy)
        
        # Statistics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_accuracy = np.mean(val_accuracies)
        current_lr = scheduler.get_last_lr()[0]
        
        scheduler.step()
        
        history.append([epoch, train_loss, val_loss, val_accuracy, current_lr])
        
        logger.log(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_accuracy:.3f}, lr={current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(h_proj.state_dict(), SAVE_DIR / "h_proj_best.pt")
            torch.save(t_proj.state_dict(), SAVE_DIR / "t_proj_best.pt")
            logger.log(f"Saved best model with val_loss={val_loss:.4f}")
    
    # Save final models and history
    torch.save(h_proj.state_dict(), SAVE_DIR / "h_proj_final.pt")
    torch.save(t_proj.state_dict(), SAVE_DIR / "t_proj_final.pt")
    
    history_df = pd.DataFrame(history, columns=["epoch", "train_loss", "val_loss", "val_accuracy", "learning_rate"])
    history_df.to_csv(SAVE_DIR / "training_history.csv", index=False)
    
    logger.log("Training completed successfully")
    return ex, te, h_proj, t_proj

# ==============================================================
# Enhanced Analysis: Layer-wise Dynamics (FIXED)
# ==============================================================
def analyze_dynamics(pairs, ex, te, h, t):
    """Enhanced analysis with additional metrics - FIXED VERSION"""
    results = []
    all_traj = {"factual": [], "hallucination": []}
    layerwise_data = {label: [] for label in ["factual", "hallucination"]}

    logger.log("Starting layer-wise dynamics analysis...")
    
    for text, truth, label in tqdm(pairs, desc="Analyzing Dynamics"):
        with torch.no_grad():
            # Get hidden states for all layers
            hidden_states = ex.get_hidden_states([text]).squeeze(0)  # [layers, hidden_dim]
            truth_embedding = te.encode_batch([truth])  # [1, truth_dim]
            
            # Project to shared space
            Hp = F.normalize(h(hidden_states), p=2, dim=-1)  # [layers, shared_dim]
            Gp = F.normalize(t(truth_embedding), p=2, dim=-1)  # [1, shared_dim]
            
            # Layer-wise alignments - FIXED: Proper dimension handling
            alignments = []
            for layer_idx in range(Hp.size(0)):
                # Each Hp[layer_idx] has shape [shared_dim]
                # Gp has shape [1, shared_dim] - we need to squeeze for cosine_similarity
                layer_embedding = Hp[layer_idx].unsqueeze(0)  # [1, shared_dim]
                cos_sim = F.cosine_similarity(layer_embedding, Gp, dim=1)
                alignments.append(cos_sim.item())

            # Dynamics metrics
            if Hp.size(0) > 1:
                deltas = Hp[1:] - Hp[:-1]  # [layers-1, shared_dim]
                velocities = torch.norm(deltas, dim=1).cpu().numpy()
                
                # Acceleration (direction consistency)
                if len(deltas) > 2:
                    accel_similarity = F.cosine_similarity(deltas[:-1], deltas[1:], dim=1)
                    acceleration = accel_similarity.mean().item()
                else:
                    acceleration = 0.0
            else:
                velocities = np.array([0.0])
                acceleration = 0.0
            
            # Enhanced metrics
            convergence_point = np.argmax(alignments)
            stability = np.std(alignments[-3:]) if len(alignments) >= 3 else np.std(alignments)
            max_alignment = np.max(alignments)
            alignment_gain = alignments[-1] - alignments[0]
            
            # Oscillation (number of direction changes)
            if len(alignments) > 2:
                second_derivative = np.diff(np.sign(np.diff(alignments)))
                oscillation = np.sum(second_derivative != 0)
            else:
                oscillation = 0
            
            results.append({
                "label": label,
                "text": text[:50] + "..." if len(text) > 50 else text,
                "truth": truth[:50] + "..." if len(truth) > 50 else truth,
                "final_alignment": alignments[-1],
                "mean_alignment": np.mean(alignments),
                "max_alignment": max_alignment,
                "convergence_layer": convergence_point,
                "stability": stability,
                "alignment_gain": alignment_gain,
                "mean_velocity": np.mean(velocities) if len(velocities) > 0 else 0.0,
                "max_velocity": np.max(velocities) if len(velocities) > 0 else 0.0,
                "mean_acceleration": acceleration,
                "oscillation": oscillation,
                "num_layers": len(alignments)
            })
            
            all_traj[label].append(alignments)
            layerwise_data[label].append(alignments)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "enhanced_dynamics_results.csv", index=False)

    
    logger.log(f"Analysis completed: {len(df)} samples processed")
    return df, all_traj, layerwise_data

# ==============================================================
# Statistical Analysis
# ==============================================================
def statistical_analysis(df, layerwise_data):
    """Comprehensive statistical analysis - FIXED VERSION"""
    logger.log("Performing statistical analysis...")
    
    # Basic group statistics
    factual_df = df[df["label"] == "factual"]
    hallucination_df = df[df["label"] == "hallucination"]
    
    stats_results = {}
    
    # T-tests for each metric
    metrics = ["final_alignment", "mean_alignment", "max_alignment", "mean_velocity", 
               "mean_acceleration", "stability", "alignment_gain", "convergence_layer", "oscillation"]
    
    for metric in metrics:
        if metric in factual_df.columns and metric in hallucination_df.columns:
            factual_vals = factual_df[metric].dropna()
            hallucination_vals = hallucination_df[metric].dropna()
            
            if len(factual_vals) > 1 and len(hallucination_vals) > 1:
                t_stat, p_value = ttest_ind(factual_vals, hallucination_vals, nan_policy='omit')
                pooled_std = np.sqrt((factual_vals.std()**2 + hallucination_vals.std()**2) / 2)
                cohens_d = (factual_vals.mean() - hallucination_vals.mean()) / pooled_std if pooled_std > 0 else 0
                
                # Ensure numeric types
                stats_results[metric] = {
                    "t_stat": float(t_stat),
                    "p_value": float(p_value),
                    "cohens_d": float(cohens_d),
                    "factual_mean": float(factual_vals.mean()),
                    "hallucination_mean": float(hallucination_vals.mean()),
                    "significant": p_value < 0.05
                }
    
    # Layer-wise statistical analysis
    layer_p_values = []
    layer_effect_sizes = []
    
    if "factual" in layerwise_data and "hallucination" in layerwise_data:
        factual_trajs = layerwise_data["factual"]
        hallucination_trajs = layerwise_data["hallucination"]
        
        if factual_trajs and hallucination_trajs:
            min_layers = min(len(factual_trajs[0]), len(hallucination_trajs[0]))
            
            for layer in range(min_layers):
                factual_vals = [traj[layer] for traj in factual_trajs]
                hallucination_vals = [traj[layer] for traj in hallucination_trajs]
                
                t_stat, p_val = ttest_ind(factual_vals, hallucination_vals)
                pooled_std = np.sqrt((np.std(factual_vals)**2 + np.std(hallucination_vals)**2) / 2)
                cohens_d = (np.mean(factual_vals) - np.mean(hallucination_vals)) / pooled_std if pooled_std > 0 else 0
                
                layer_p_values.append(float(p_val))
                layer_effect_sizes.append(float(cohens_d))
    
    # Compile results
    stats_summary = {
        "metric_ttests": pd.DataFrame(stats_results).T if stats_results else pd.DataFrame(),
        "layerwise_significance": {
            "p_values": layer_p_values,
            "effect_sizes": layer_effect_sizes,
            "significant_layers": np.sum(np.array(layer_p_values) < 0.05) if layer_p_values else 0
        },
        "sample_sizes": {
            "factual": len(factual_df),
            "hallucination": len(hallucination_df)
        }
    }
    
    # Ensure numeric types in the dataframe
    if not stats_summary["metric_ttests"].empty:
        numeric_columns = ['t_stat', 'p_value', 'cohens_d', 'factual_mean', 'hallucination_mean']
        for col in numeric_columns:
            if col in stats_summary["metric_ttests"].columns:
                stats_summary["metric_ttests"][col] = pd.to_numeric(stats_summary["metric_ttests"][col], errors='coerce')
        
        stats_summary["metric_ttests"].to_csv(RESULTS_DIR / "statistical_significance.csv")
    
    # Create summary report
    with open(RESULTS_DIR / "statistical_summary.txt", "w") as f:
        f.write("LAYER-WISE SEMANTIC DYNAMICS - STATISTICAL SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Sample sizes: Factual={len(factual_df)}, Hallucination={len(hallucination_df)}\n\n")
        
        if stats_results:
            f.write("METRIC COMPARISONS (Factual vs Hallucination):\n")
            f.write("-" * 50 + "\n")
            for metric, results in stats_results.items():
                sig_flag = "***" if results["p_value"] < 0.001 else "**" if results["p_value"] < 0.01 else "*" if results["p_value"] < 0.05 else ""
                f.write(f"{metric:20}: p={results['p_value']:.4f}{sig_flag}, d={results['cohens_d']:.3f}\n")
        
        if layer_p_values:
            f.write(f"\nLAYER-WISE SIGNIFICANCE: {stats_summary['layerwise_significance']['significant_layers']}/{len(layer_p_values)} layers significant (p < 0.05)\n")
    
    logger.log("Statistical analysis completed")
    return stats_summary

# ==============================================================
# FIXED Visualization Functions (DEFINED BEFORE MAIN)
# ==============================================================
def plot_convergence(all_traj, stats_summary):
    """Fixed convergence plot with proper error handling"""
    logger.log("Generating convergence plots...")
    
    if not all_traj or ("factual" not in all_traj and "hallucination" not in all_traj):
        logger.log("Warning: No trajectory data available for convergence plot")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Mean trajectories with confidence intervals
    if "factual" in all_traj and all_traj["factual"]:
        factual_trajs = np.array(all_traj["factual"])
        factual_mean = np.mean(factual_trajs, axis=0)
        factual_std = np.std(factual_trajs, axis=0)
        layers = range(len(factual_mean))
        
        axes[0,0].plot(layers, factual_mean, 'g-', linewidth=2, label='Factual')
        axes[0,0].fill_between(layers, factual_mean - factual_std, factual_mean + factual_std, 
                              alpha=0.2, color='green')
    
    if "hallucination" in all_traj and all_traj["hallucination"]:
        hallucination_trajs = np.array(all_traj["hallucination"])
        hallucination_mean = np.mean(hallucination_trajs, axis=0)
        hallucination_std = np.std(hallucination_trajs, axis=0)
        layers = range(len(hallucination_mean))
        
        axes[0,0].plot(layers, hallucination_mean, 'r-', linewidth=2, label='Hallucination')
        axes[0,0].fill_between(layers, hallucination_mean - hallucination_std, 
                              hallucination_mean + hallucination_std, alpha=0.2, color='red')
    
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('Alignment with Truth')
    axes[0,0].set_title('Layer-wise Semantic Alignment Trajectories')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Layer-wise significance
    if (stats_summary and 'layerwise_significance' in stats_summary and 
        stats_summary['layerwise_significance']['p_values']):
        
        p_values = stats_summary['layerwise_significance']['p_values']
        layers = range(len(p_values))
        
        # Plot p-values
        axes[0,1].plot(layers, p_values, 'b-', linewidth=2, label='p-value')
        axes[0,1].axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='p=0.05 threshold')
        axes[0,1].set_yscale('log')
        axes[0,1].set_xlabel('Layer')
        axes[0,1].set_ylabel('p-value (log scale)')
        axes[0,1].set_title('Layer-wise Statistical Significance')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Effect sizes
    if (stats_summary and 'layerwise_significance' in stats_summary and 
        stats_summary['layerwise_significance']['effect_sizes']):
        
        effect_sizes = stats_summary['layerwise_significance']['effect_sizes']
        layers = range(len(effect_sizes))
        
        axes[1,0].bar(layers, effect_sizes, color=['green' if es > 0 else 'red' for es in effect_sizes], alpha=0.7)
        axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1,0].set_xlabel('Layer')
        axes[1,0].set_ylabel("Cohen's d")
        axes[1,0].set_title('Layer-wise Effect Sizes\n(Positive = Factual > Hallucination)')
        axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Final alignment distribution
    if "factual" in all_traj and "hallucination" in all_traj:
        factual_final = [traj[-1] for traj in all_traj["factual"]] if all_traj["factual"] else []
        hallucination_final = [traj[-1] for traj in all_traj["hallucination"]] if all_traj["hallucination"] else []
        
        if factual_final and hallucination_final:
            axes[1,1].hist(factual_final, bins=20, alpha=0.7, color='green', label='Factual', density=True)
            axes[1,1].hist(hallucination_final, bins=20, alpha=0.7, color='red', label='Hallucination', density=True)
            axes[1,1].set_xlabel('Final Alignment Score')
            axes[1,1].set_ylabel('Density')
            axes[1,1].set_title('Distribution of Final Alignment Scores')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "convergence_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.log("Convergence plots generated successfully")

def plot_velocity_acceleration(df):
    """Enhanced velocity and acceleration plots - FIXED VERSION"""
    logger.log("Generating velocity and acceleration plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Velocity distribution
    if "mean_velocity" in df.columns:
        sns.violinplot(data=df, x="label", y="mean_velocity", hue="label", 
                       palette={"factual": "green", "hallucination": "red"}, 
                       ax=axes[0,0], legend=False)
        axes[0,0].set_title("Semantic Velocity Distribution")
        axes[0,0].set_ylabel("Mean Velocity")
    
    # Acceleration distribution
    if "mean_acceleration" in df.columns:
        sns.violinplot(data=df, x="label", y="mean_acceleration", hue="label",
                       palette={"factual": "green", "hallucination": "red"}, 
                       ax=axes[0,1], legend=False)
        axes[0,1].set_title("Semantic Acceleration Distribution")
        axes[0,1].set_ylabel("Mean Acceleration")
    
    # Alignment gain
    if "alignment_gain" in df.columns:
        sns.violinplot(data=df, x="label", y="alignment_gain", hue="label",
                       palette={"factual": "green", "hallucination": "red"}, 
                       ax=axes[1,0], legend=False)
        axes[1,0].set_title("Alignment Gain (Final - Initial)")
        axes[1,0].set_ylabel("Alignment Gain")
    
    # Convergence layer
    if "convergence_layer" in df.columns:
        sns.violinplot(data=df, x="label", y="convergence_layer", hue="label",
                       palette={"factual": "green", "hallucination": "red"}, 
                       ax=axes[1,1], legend=False)
        axes[1,1].set_title("Layer of Maximum Alignment")
        axes[1,1].set_ylabel("Convergence Layer")
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "enhanced_velocity_acceleration.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.log("Velocity and acceleration plots generated successfully")

# ==============================================================
# Fixed Main Execution
# ==============================================================
def main():
    """Enhanced main execution with comprehensive analysis - FIXED VERSION"""
    logger.log("Starting Layer-wise Semantic Dynamics Analysis")
    logger.log(f"Configuration: {config}")
    logger.log(f"Device: {DEVICE}")
    
    try:
        # Step 1: Build dataset
        logger.log("Step 1: Building dataset pairs...")
        pairs = build_pairs(config.num_pairs)
        
        if not pairs:
            logger.log("ERROR: No pairs generated. Exiting.")
            return
        
        # Step 2: Train projection heads
        logger.log("Step 2: Training projection heads...")
        ex, te, h_proj, t_proj = train(pairs)
        
        # Step 3: Analyze dynamics
        logger.log("Step 3: Analyzing layer-wise dynamics...")
        df, all_traj, layerwise_data = analyze_dynamics(pairs, ex, te, h_proj, t_proj)
        
        # Step 4: Statistical analysis
        logger.log("Step 4: Performing statistical analysis...")
        stats_summary = statistical_analysis(df, layerwise_data)
        
        # Step 5: Visualization
        logger.log("Step 5: Generating visualizations...")
        plot_convergence(all_traj, stats_summary)
        plot_velocity_acceleration(df)
        
        # Final summary
        logger.log("ANALYSIS COMPLETED SUCCESSFULLY")
        logger.log("=" * 50)
        logger.log("KEY FINDINGS:")
        logger.log(f"- Total samples analyzed: {len(df)}")
        logger.log(f"- Factual samples: {len(df[df['label']=='factual'])}")
        logger.log(f"- Hallucination samples: {len(df[df['label']=='hallucination'])}")
        
        # Print key statistical results
        if not stats_summary['metric_ttests'].empty:
            # Ensure we have numeric p-values
            stats_summary['metric_ttests']['p_value'] = pd.to_numeric(stats_summary['metric_ttests']['p_value'], errors='coerce')
            
            sig_metrics = stats_summary['metric_ttests'][stats_summary['metric_ttests']['significant']]
            logger.log(f"- Significant metrics: {len(sig_metrics)}/{len(stats_summary['metric_ttests'])}")
            
            # Print top 3 most significant metrics
            if not stats_summary['metric_ttests'].empty:
                top_metrics = stats_summary['metric_ttests'].nsmallest(3, 'p_value')
                logger.log("TOP SIGNIFICANT METRICS:")
                for metric, row in top_metrics.iterrows():
                    logger.log(f"  {metric}: p={row['p_value']:.6f}, d={row['cohens_d']:.3f}")
        
        if stats_summary['layerwise_significance']['p_values']:
            logger.log(f"- Significant layers: {stats_summary['layerwise_significance']['significant_layers']}/{len(stats_summary['layerwise_significance']['p_values'])}")
        
        # Save final summary
        summary_report = f"""
LAYER-WISE SEMANTIC DYNAMICS - FINAL REPORT
===========================================

Dataset:
- Total pairs: {len(pairs)}
- Factual: {len(df[df['label']=='factual'])}
- Hallucination: {len(df[df['label']=='hallucination'])}

Key Findings:"""
        
        if not stats_summary['metric_ttests'].empty:
            sig_metrics = stats_summary['metric_ttests'][stats_summary['metric_ttests']['significant']]
            summary_report += f"""
- Significant metrics: {len(sig_metrics)}/{len(stats_summary['metric_ttests'])}
- Significant layers: {stats_summary['layerwise_significance']['significant_layers']}/{len(stats_summary['layerwise_significance']['p_values'])}

Most Significant Differences:"""
            if not stats_summary['metric_ttests'].empty:
                top_metrics = stats_summary['metric_ttests'].nsmallest(3, 'p_value')
                for metric, row in top_metrics.iterrows():
                    summary_report += f"\n- {metric}: p={row['p_value']:.6f}, Cohen's d={row['cohens_d']:.3f}"
        
        with open(RESULTS_DIR / "final_report.txt", "w") as f:
            f.write(summary_report)
        
        print("\n" + "="*60)
        print("LAYER-WISE SEMANTIC DYNAMICS ANALYSIS COMPLETE")
        print("="*60)
        print(f"Results saved to: {SAVE_DIR}")
        print(f"Plots saved to: {PLOT_DIR}")
        print(f"Analysis results: {RESULTS_DIR}")
        print("\nSummary Statistics:")
        summary_stats = df.groupby("label")[["final_alignment", "mean_alignment", "mean_velocity", "mean_acceleration"]].mean().round(4)
        print(summary_stats)
        
        # Print key insights
        print("\nKEY INSIGHTS:")
        if not stats_summary['metric_ttests'].empty:
            factual_final = df[df['label']=='factual']['final_alignment'].mean()
            hallucination_final = df[df['label']=='hallucination']['final_alignment'].mean()
            print(f"- Final alignment: Factual ({factual_final:.3f}) vs Hallucination ({hallucination_final:.3f})")
            
            if factual_final > hallucination_final:
                print("  → Factual content shows BETTER final alignment with truth")
            else:
                print("  → Hallucinated content shows BETTER final alignment with truth (unexpected)")
        
    except Exception as e:
        logger.log(f"ERROR in main execution: {e}")
        import traceback
        logger.log(f"Traceback: {traceback.format_exc()}")
        print(f"Error occurred: {e}")
        print("Check the log file for details.")

if __name__ == "__main__":
    main()