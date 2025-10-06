import random
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset
import datasets
from ..utils.helpers import logger
from ..core.config import LayerwiseSemanticDynamicsConfig

class DataManager:
    """
    Comprehensive data manager supporting multiple dataset sources.
    Handles synthetic data generation, TruthfulQA loading, and class balancing.
    """
    def __init__(self, config: LayerwiseSemanticDynamicsConfig):
        self.config = config
        
    def _load_synthetic_pairs(self) -> List[Tuple[str, str, str]]:
        """Generate comprehensive synthetic dataset with factual and hallucination pairs."""
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
        """Load TruthfulQA dataset from HuggingFace."""
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
        """Build comprehensive dataset from multiple sources with class balancing."""
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
    """PyTorch Dataset for text-truth pairs."""
    
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