# ==============================================================
# Layer-wise Semantic Dynamics (LSD)
# Complete Fixed Implementation
# ==============================================================
# !pip install -q transformers sentence-transformers datasets torch tqdm matplotlib scikit-learn scipy seaborn

import os, math, random
# Add these imports to your existing imports section
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import json
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
    num_pairs: int = 500
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
# ADD AFTER YOUR EXISTING UTILITY FUNCTIONS
# ==============================================================

# ==============================================================
# COMPREHENSIVE EVALUATION FUNCTIONS - ADD THIS SECTION
# ==============================================================

class LSDDetector:
    """Complete LSD-based hallucination detector with evaluation metrics"""
    
    def __init__(self, ex, te, h_proj, t_proj):
        self.ex = ex
        self.te = te
        self.h_proj = h_proj
        self.t_proj = t_proj
        
    def extract_features(self, texts, truths):
        """Extract LSD trajectory features for classification"""
        features = []
        
        for text, truth in zip(texts, truths):
            with torch.no_grad():
                # Get hidden states and project
                hidden_states = self.ex.get_hidden_states([text]).squeeze(0)
                truth_embedding = self.te.encode_batch([truth])
                
                Hp = F.normalize(self.h_proj(hidden_states), p=2, dim=-1)
                Gp = F.normalize(self.t_proj(truth_embedding), p=2, dim=-1)
                
                # Compute alignment trajectory
                alignments = []
                for layer_idx in range(Hp.size(0)):
                    layer_embedding = Hp[layer_idx].unsqueeze(0)
                    cos_sim = F.cosine_similarity(layer_embedding, Gp, dim=1)
                    alignments.append(cos_sim.item())
                
                # Extract comprehensive features
                feature_vector = [
                    alignments[-1],  # final_alignment
                    np.mean(alignments),  # mean_alignment
                    np.max(alignments),  # max_alignment
                    np.std(alignments[-3:]) if len(alignments) >= 3 else np.std(alignments),  # stability
                    alignments[-1] - alignments[0],  # alignment_gain
                    np.argmax(alignments),  # convergence_layer
                ]
                
                # Add velocity features if possible
                if Hp.size(0) > 1:
                    deltas = Hp[1:] - Hp[:-1]
                    velocities = torch.norm(deltas, dim=1).cpu().numpy()
                    feature_vector.extend([
                        np.mean(velocities),  # mean_velocity
                        np.max(velocities),  # max_velocity
                    ])
                    
                    if len(deltas) > 2:
                        accel_similarity = F.cosine_similarity(deltas[:-1], deltas[1:], dim=1)
                        feature_vector.append(accel_similarity.mean().item())  # mean_acceleration
                    else:
                        feature_vector.append(0.0)
                else:
                    feature_vector.extend([0.0, 0.0, 0.0])
                
                features.append(feature_vector)
        
        return np.array(features)

def comprehensive_evaluation(df, layerwise_data, ex, te, h_proj, t_proj):
    """Comprehensive evaluation with all metrics and benchmarking"""
    logger.log("Starting comprehensive evaluation...")
    
    # Initialize LSD detector
    lsd_detector = LSDDetector(ex, te, h_proj, t_proj)
    
    # Prepare data
    texts = df['text'].tolist()
    truths = df['truth'].tolist()
    labels = (df['label'] == 'factual').astype(int).tolist()  # 1 for factual, 0 for hallucination
    
    # Extract features
    logger.log("Extracting LSD features...")
    X = lsd_detector.extract_features(texts, truths)
    y = np.array(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train classifiers
    classifiers = {
        'LSD_LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'LSD_RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
    }
    
    results = {}
    
    for clf_name, clf in classifiers.items():
        logger.log(f"Training {clf_name}...")
        
        # Train classifier
        clf.fit(X_train, y_train)
        
        # Predictions
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]  # Probability of factual class
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        accuracy = accuracy_score(y_test, y_pred)
        auroc = roc_auc_score(y_test, y_pred_proba)
        
        # Specificity (True Negative Rate)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Precision-Recall AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        prauc = auc(recall_curve, precision_curve)
        
        # F2 Score (emphasizes recall)
        f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
        
        results[clf_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f2': f2,
            'accuracy': accuracy,
            'specificity': specificity,
            'auroc': auroc,
            'prauc': prauc,
            'confusion_matrix': (tn, fp, fn, tp),
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        logger.log(f"{clf_name} - F1: {f1:.4f}, AUC-ROC: {auroc:.4f}, Specificity: {specificity:.4f}")
    
    # Add simple threshold-based LSD method
    logger.log("Evaluating simple LSD threshold method...")
    simple_scores = X_test[:, 0]  # Use final_alignment as score
    simple_pred = (simple_scores > 0).astype(int)
    simple_pred_proba = 1 / (1 + np.exp(-simple_scores * 10))  # Sigmoid scaling
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, simple_pred, average='binary')
    accuracy = accuracy_score(y_test, simple_pred)
    auroc = roc_auc_score(y_test, simple_pred_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_test, simple_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, simple_pred_proba)
    prauc = auc(recall_curve, precision_curve)
    f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
    
    results['LSD_SimpleThreshold'] = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
        'accuracy': accuracy,
        'specificity': specificity,
        'auroc': auroc,
        'prauc': prauc,
        'confusion_matrix': (tn, fp, fn, tp),
        'y_true': y_test,
        'y_pred': simple_pred,
        'y_pred_proba': simple_pred_proba
    }
    
    return results, X, y

def plot_comprehensive_metrics(results):
    """Plot comprehensive evaluation metrics"""
    logger.log("Generating comprehensive metrics plots...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract method names and metrics
    methods = list(results.keys())
    metrics = ['precision', 'recall', 'f1', 'specificity', 'auroc', 'prauc']
    metric_names = ['Precision', 'Recall', 'F1-Score', 'Specificity', 'AUC-ROC', 'PR-AUC']
    
    # Plot 1: Main metrics comparison
    metric_values = {metric: [results[method][metric] for method in methods] for metric in metrics}
    
    x = np.arange(len(methods))
    width = 0.12
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        axes[0,0].bar(x + i*width, metric_values[metric], width, label=metric_name, alpha=0.8)
    
    axes[0,0].set_xlabel('Methods')
    axes[0,0].set_ylabel('Score')
    axes[0,0].set_title('Comprehensive Metrics Comparison')
    axes[0,0].set_xticks(x + width*2.5)
    axes[0,0].set_xticklabels(methods, rotation=45)
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: ROC Curves
    for method_name, result in results.items():
        fpr, tpr, _ = roc_curve(result['y_true'], result['y_pred_proba'])
        axes[0,1].plot(fpr, tpr, label=f'{method_name} (AUC = {result["auroc"]:.3f})', linewidth=2)
    
    axes[0,1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].set_title('ROC Curves')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Precision-Recall Curves
    for method_name, result in results.items():
        precision_curve, recall_curve, _ = precision_recall_curve(result['y_true'], result['y_pred_proba'])
        axes[0,2].plot(recall_curve, precision_curve, 
                      label=f'{method_name} (AUC = {result["prauc"]:.3f})', linewidth=2)
    
    # Add no-skill line
    factual_ratio = np.mean([result['y_true'] for result in results.values()][0])
    axes[0,2].axhline(y=factual_ratio, color='k', linestyle='--', alpha=0.5, label='No Skill')
    axes[0,2].set_xlabel('Recall')
    axes[0,2].set_ylabel('Precision')
    axes[0,2].set_title('Precision-Recall Curves')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot confusion matrices for up to 3 methods
    for i, (method_name, result) in enumerate(list(results.items())[:3]):
        tn, fp, fn, tp = result['confusion_matrix']
        cm = np.array([[tn, fp], [fn, tp]])
        
        im = axes[1, i].imshow(cm, cmap='Blues', alpha=0.8)
        axes[1, i].set_xticks([0, 1])
        axes[1, i].set_yticks([0, 1])
        axes[1, i].set_xticklabels(['Hallucination', 'Factual'])
        axes[1, i].set_yticklabels(['Hallucination', 'Factual'])
        axes[1, i].set_xlabel('Predicted')
        axes[1, i].set_ylabel('Actual')
        axes[1, i].set_title(f'{method_name}\nConfusion Matrix')
        
        # Add text annotations
        for i_val in range(2):
            for j in range(2):
                axes[1, i].text(j, i_val, f'{cm[i_val, j]}', 
                              ha='center', va='center', 
                              color='white' if cm[i_val, j] > cm.max()/2 else 'black',
                              fontweight='bold')
        
        plt.colorbar(im, ax=axes[1, i], shrink=0.6)
    
    # Hide empty subplots
    for i in range(len(results), 3):
        if i < 3:  # Only hide if within bounds
            axes[1, i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "comprehensive_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()

def generate_evaluation_report(results, df):
    """Generate comprehensive evaluation report"""
    logger.log("Generating evaluation report...")
    
    report = {
        'dataset_info': {
            'total_samples': len(df),
            'factual_samples': len(df[df['label'] == 'factual']),
            'hallucination_samples': len(df[df['label'] == 'hallucination']),
            'class_balance': len(df[df['label'] == 'factual']) / len(df)
        },
        'methods': {},
        'best_performing': {},
        'recommendations': []
    }
    
    # Add method results
    for method_name, result in results.items():
        report['methods'][method_name] = {
            'precision': float(result['precision']),
            'recall': float(result['recall']),
            'f1_score': float(result['f1']),
            'f2_score': float(result['f2']),
            'accuracy': float(result['accuracy']),
            'specificity': float(result['specificity']),
            'auc_roc': float(result['auroc']),
            'pr_auc': float(result['prauc']),
            'confusion_matrix': {
                'true_negatives': int(result['confusion_matrix'][0]),
                'false_positives': int(result['confusion_matrix'][1]),
                'false_negatives': int(result['confusion_matrix'][2]),
                'true_positives': int(result['confusion_matrix'][3])
            }
        }
    
    # Find best performing method for each metric
    metrics = ['f1', 'auroc', 'precision', 'recall', 'specificity']
    for metric in metrics:
        best_method = max(results.keys(), key=lambda x: results[x][metric])
        report['best_performing'][metric] = {
            'method': best_method,
            'score': float(results[best_method][metric])
        }
    
    # Generate recommendations
    best_f1_method = report['best_performing']['f1']['method']
    best_auc_method = report['best_performing']['auroc']['method']
    
    report['recommendations'] = [
        f"Best overall performance: {best_f1_method} (F1: {report['best_performing']['f1']['score']:.3f})",
        f"Best ranking performance: {best_auc_method} (AUC-ROC: {report['best_performing']['auroc']['score']:.3f})",
        "LSD_SimpleThreshold is most interpretable but may have lower performance",
        "LSD_RandomForest provides feature importance insights",
        "Consider deployment requirements: precision vs recall trade-offs"
    ]
    
    # Save report
    with open(RESULTS_DIR / "comprehensive_evaluation_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("="*70)
    
    for method_name, result in results.items():
        print(f"\n{method_name}:")
        print(f"  F1: {result['f1']:.4f} | AUC-ROC: {result['auroc']:.4f} | Specificity: {result['specificity']:.4f}")
        print(f"  Precision: {result['precision']:.4f} | Recall: {result['recall']:.4f}")
    
    print(f"\nBest F1: {report['best_performing']['f1']['method']} ({report['best_performing']['f1']['score']:.4f})")
    print(f"Best AUC-ROC: {report['best_performing']['auroc']['method']} ({report['best_performing']['auroc']['score']:.4f})")
    
    return report

def plot_comprehensive_metrics(results):
    """Plot comprehensive evaluation metrics"""
    logger.log("Generating comprehensive metrics plots...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract method names and metrics
    methods = list(results.keys())
    metrics = ['precision', 'recall', 'f1', 'specificity', 'auroc', 'prauc']
    metric_names = ['Precision', 'Recall', 'F1-Score', 'Specificity', 'AUC-ROC', 'PR-AUC']
    
    # Plot 1: Main metrics comparison
    metric_values = {metric: [results[method][metric] for method in methods] for metric in metrics}
    
    x = np.arange(len(methods))
    width = 0.12
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        axes[0,0].bar(x + i*width, metric_values[metric], width, label=metric_name, alpha=0.8)
    
    axes[0,0].set_xlabel('Methods')
    axes[0,0].set_ylabel('Score')
    axes[0,0].set_title('Comprehensive Metrics Comparison')
    axes[0,0].set_xticks(x + width*2.5)
    axes[0,0].set_xticklabels(methods, rotation=45)
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: ROC Curves
    for method_name, result in results.items():
        fpr, tpr, _ = roc_curve(result['y_true'], result['y_pred_proba'])
        axes[0,1].plot(fpr, tpr, label=f'{method_name} (AUC = {result["auroc"]:.3f})', linewidth=2)
    
    axes[0,1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].set_title('ROC Curves')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Precision-Recall Curves
    for method_name, result in results.items():
        precision_curve, recall_curve, _ = precision_recall_curve(result['y_true'], result['y_pred_proba'])
        axes[0,2].plot(recall_curve, precision_curve, 
                      label=f'{method_name} (AUC = {result["prauc"]:.3f})', linewidth=2)
    
    # Add no-skill line
    factual_ratio = np.mean([result['y_true'] for result in results.values()][0])
    axes[0,2].axhline(y=factual_ratio, color='k', linestyle='--', alpha=0.5, label='No Skill')
    axes[0,2].set_xlabel('Recall')
    axes[0,2].set_ylabel('Precision')
    axes[0,2].set_title('Precision-Recall Curves')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot confusion matrices for up to 3 methods
    for i, (method_name, result) in enumerate(list(results.items())[:3]):
        tn, fp, fn, tp = result['confusion_matrix']
        cm = np.array([[tn, fp], [fn, tp]])
        
        im = axes[1, i].imshow(cm, cmap='Blues', alpha=0.8)
        axes[1, i].set_xticks([0, 1])
        axes[1, i].set_yticks([0, 1])
        axes[1, i].set_xticklabels(['Hallucination', 'Factual'])
        axes[1, i].set_yticklabels(['Hallucination', 'Factual'])
        axes[1, i].set_xlabel('Predicted')
        axes[1, i].set_ylabel('Actual')
        axes[1, i].set_title(f'{method_name}\nConfusion Matrix')
        
        # Add text annotations
        for i_val in range(2):
            for j in range(2):
                axes[1, i].text(j, i_val, f'{cm[i_val, j]}', 
                              ha='center', va='center', 
                              color='white' if cm[i_val, j] > cm.max()/2 else 'black',
                              fontweight='bold')
        
        plt.colorbar(im, ax=axes[1, i], shrink=0.6)
    
    # Hide empty subplots
    for i in range(len(results), 3):
        axes[1, i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "comprehensive_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()

def generate_evaluation_report(results, df):
    """Generate comprehensive evaluation report"""
    logger.log("Generating evaluation report...")
    
    report = {
        'dataset_info': {
            'total_samples': len(df),
            'factual_samples': len(df[df['label'] == 'factual']),
            'hallucination_samples': len(df[df['label'] == 'hallucination']),
            'class_balance': len(df[df['label'] == 'factual']) / len(df)
        },
        'methods': {},
        'best_performing': {},
        'recommendations': []
    }
    
    # Add method results
    for method_name, result in results.items():
        report['methods'][method_name] = {
            'precision': float(result['precision']),
            'recall': float(result['recall']),
            'f1_score': float(result['f1']),
            'f2_score': float(result['f2']),
            'accuracy': float(result['accuracy']),
            'specificity': float(result['specificity']),
            'auc_roc': float(result['auroc']),
            'pr_auc': float(result['prauc']),
            'confusion_matrix': {
                'true_negatives': int(result['confusion_matrix'][0]),
                'false_positives': int(result['confusion_matrix'][1]),
                'false_negatives': int(result['confusion_matrix'][2]),
                'true_positives': int(result['confusion_matrix'][3])
            }
        }
    
    # Find best performing method for each metric
    metrics = ['f1', 'auroc', 'precision', 'recall', 'specificity']
    for metric in metrics:
        best_method = max(results.keys(), key=lambda x: results[x][metric])
        report['best_performing'][metric] = {
            'method': best_method,
            'score': float(results[best_method][metric])
        }
    
    # Generate recommendations
    best_f1_method = report['best_performing']['f1']['method']
    best_auc_method = report['best_performing']['auroc']['method']
    
    report['recommendations'] = [
        f"Best overall performance: {best_f1_method} (F1: {report['best_performing']['f1']['score']:.3f})",
        f"Best ranking performance: {best_auc_method} (AUC-ROC: {report['best_performing']['auroc']['score']:.3f})",
        "LSD_SimpleThreshold is most interpretable but may have lower performance",
        "LSD_RandomForest provides feature importance insights",
        "Consider deployment requirements: precision vs recall trade-offs"
    ]
    
    # Save report
    with open(RESULTS_DIR / "comprehensive_evaluation_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("="*70)
    
    for method_name, result in results.items():
        print(f"\n{method_name}:")
        print(f"  F1: {result['f1']:.4f} | AUC-ROC: {result['auroc']:.4f} | Specificity: {result['specificity']:.4f}")
        print(f"  Precision: {result['precision']:.4f} | Recall: {result['recall']:.4f}")
    
    print(f"\nBest F1: {report['best_performing']['f1']['method']} ({report['best_performing']['f1']['score']:.4f})")
    print(f"Best AUC-ROC: {report['best_performing']['auroc']['method']} ({report['best_performing']['auroc']['score']:.4f})")
    
    return report

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

        ("Binary code uses ones and zeros.", "Digital computers process information in binary format.", "factual"),
        ("WiFi transmits data wirelessly.", "Wireless fidelity enables internet connectivity without cables.", "factual"),
        ("USB devices connect to computers.", "Universal Serial Bus allows peripheral device connections.", "factual"),
        ("RAM stores temporary data.", "Random Access Memory holds data during active use.", "factual"),
        ("CPUs process computer instructions.", "The central processing unit executes program operations.", "factual"),
        ("Keyboards are input devices.", "Computer keyboards allow text and command entry.", "factual"),
        ("HTML structures web pages.", "HyperText Markup Language defines webpage content.", "factual"),
        ("Browsers display web content.", "Web browsers render and display internet pages.", "factual"),
        ("Email sends digital messages.", "Electronic mail transmits messages over networks.", "factual"),
        ("Bluetooth connects devices short-range.", "Bluetooth technology enables wireless device pairing.", "factual"),
        ("Firewalls protect networks.", "Network firewalls filter unauthorized access attempts.", "factual"),
        ("Cloud storage saves data remotely.", "Cloud services store files on remote servers.", "factual"),
        ("Encryption secures data.", "Cryptographic methods protect information from unauthorized access.", "factual"),
        ("Routers direct network traffic.", "Network routers forward data between devices.", "factual"),
        ("Operating systems manage hardware.", "OS software coordinates computer resources.", "factual"),
        ("Pixels form digital images.", "Picture elements compose digital displays.", "factual"),
        ("Cache speeds up data access.", "Cached data provides faster retrieval times.", "factual"),
        ("Algorithms solve computational problems.", "Step-by-step procedures process information systematically.", "factual"),
        ("Databases store structured information.", "Database systems organize and manage data collections.", "factual"),
        ("APIs enable software communication.", "Application Programming Interfaces allow program interaction.", "factual"),
        ("SSD drives use flash memory.", "Solid state drives store data electronically.", "factual"),
        ("Monitors display visual output.", "Computer screens present graphical information.", "factual"),
        ("Touchscreens detect finger contact.", "Touch-sensitive displays register physical interaction.", "factual"),
        ("Servers host online services.", "Server computers provide resources to client devices.", "factual"),
        ("Backup copies protect data.", "Data backups prevent information loss.", "factual"),
        ("Malware damages computer systems.", "Malicious software harms digital devices.", "factual"),
        ("VPNs create secure connections.", "Virtual Private Networks encrypt internet traffic.", "factual"),
        ("Cookies track website data.", "Browser cookies store user information.", "factual"),
        ("Updates patch software vulnerabilities.", "Software updates fix security flaws.", "factual"),
        ("Bandwidth measures data transfer rate.", "Network bandwidth indicates transmission capacity.", "factual"),
        ("Streaming delivers real-time content.", "Media streaming sends data continuously.", "factual"),
        ("Search engines index web content.", "Search tools catalog internet information.", "factual"),
        ("PDF preserves document formatting.", "Portable Document Format maintains layout integrity.", "factual"),
        ("Compression reduces file size.", "Data compression makes files smaller.", "factual"),
        ("DNS translates domain names.", "Domain Name System converts URLs to IP addresses.", "factual"),
        ("Antivirus software detects threats.", "Security programs identify malicious code.", "factual"),
        ("Git tracks code changes.", "Version control systems monitor file modifications.", "factual"),
        ("URLs specify web addresses.", "Uniform Resource Locators identify online resources.", "factual"),
        ("Metadata describes data properties.", "Metadata provides information about other data.", "factual"),
        ("Machine learning enables pattern recognition.", "ML algorithms identify patterns in data.", "factual"),
        ("QR codes store scannable information.", "Quick Response codes encode digital data.", "factual"),
        ("Hashtags categorize social content.", "Hash symbols organize posts by topic.", "factual"),
        ("Spam filters block unwanted email.", "Email filters prevent junk messages.", "factual"),
        ("Screenshots capture screen images.", "Screen captures save display contents.", "factual"),
        ("Notifications alert users to events.", "Push notifications deliver timely messages.", "factual"),
        ("Download transfers files locally.", "Downloading copies remote files to devices.", "factual"),
        ("Upload sends files remotely.", "Uploading transfers local files to servers.", "factual"),
        ("Hyperlinks connect web pages.", "Links enable navigation between documents.", "factual"),
        ("Authentication verifies user identity.", "Login systems confirm user credentials.", "factual"),
        ("Syntax defines programming rules.", "Code syntax specifies language structure.", "factual"),
        
        # Hallucination
        ("Binary code uses letters only.", "Binary systems use 0 and 1, not letters.", "hallucination"),
        ("WiFi requires physical cables.", "WiFi is wireless and doesn't need cables.", "hallucination"),
        ("USB devices charge computers.", "USB devices connect to computers, not charge them.", "hallucination"),
        ("RAM stores permanent data.", "RAM is temporary; storage drives hold permanent data.", "hallucination"),
        ("CPUs display graphics only.", "CPUs process all instructions, not just graphics.", "hallucination"),
        ("Keyboards produce sound output.", "Keyboards are input devices, not audio output.", "hallucination"),
        ("HTML is a programming language.", "HTML is a markup language, not a programming language.", "hallucination"),
        ("Browsers create web content.", "Browsers display content, they don't create it.", "hallucination"),
        ("Email requires physical delivery.", "Email is digital and requires no physical delivery.", "hallucination"),
        ("Bluetooth works across miles.", "Bluetooth has short-range, typically under 100 meters.", "hallucination"),
        ("Firewalls generate electricity.", "Firewalls are security software, not power generators.", "hallucination"),
        ("Cloud storage is physical boxes.", "Cloud storage uses remote servers, not physical boxes.", "hallucination"),
        ("Encryption makes data visible.", "Encryption obscures data for security.", "hallucination"),
        ("Routers consume internet data.", "Routers direct traffic, they don't consume data.", "hallucination"),
        ("Operating systems are hardware.", "Operating systems are software, not hardware.", "hallucination"),
        ("Pixels are physical paint.", "Pixels are digital light elements, not paint.", "hallucination"),
        ("Cache deletes all data.", "Cache stores data temporarily for faster access.", "hallucination"),
        ("Algorithms create random chaos.", "Algorithms follow systematic procedures.", "hallucination"),
        ("Databases delete information automatically.", "Databases store and preserve information.", "hallucination"),
        ("APIs prevent software communication.", "APIs enable communication between programs.", "hallucination"),
        ("SSD drives use magnetic tape.", "SSDs use flash memory, not magnetic tape.", "hallucination"),
        ("Monitors generate computer power.", "Monitors display output, they don't generate power.", "hallucination"),
        ("Touchscreens work with gloves always.", "Many touchscreens don't detect gloved fingers.", "hallucination"),
        ("Servers store data on paper.", "Servers use digital storage, not paper.", "hallucination"),
        ("Backups always fail.", "Backups are designed to prevent data loss.", "hallucination"),
        ("Malware improves computer speed.", "Malware harms systems, it doesn't improve them.", "hallucination"),
        ("VPNs expose your location.", "VPNs hide your location and encrypt traffic.", "hallucination"),
        ("Cookies are edible files.", "Browser cookies are data files, not food.", "hallucination"),
        ("Updates introduce only bugs.", "Updates typically fix bugs and add security.", "hallucination"),
        ("Bandwidth measures physical weight.", "Bandwidth measures data transfer rate.", "hallucination"),
        ("Streaming downloads entire files first.", "Streaming delivers content continuously in real-time.", "hallucination"),
        ("Search engines create web content.", "Search engines index existing content.", "hallucination"),
        ("PDF files can self-edit.", "PDFs preserve formatting but don't self-edit.", "hallucination"),
        ("Compression enlarges file size.", "Compression reduces file size.", "hallucination"),
        ("DNS creates domain names.", "DNS translates existing domain names to IPs.", "hallucination"),
        ("Antivirus creates viruses.", "Antivirus software detects and removes threats.", "hallucination"),
        ("Git deletes all code permanently.", "Git tracks and preserves code history.", "hallucination"),
        ("URLs are physical addresses.", "URLs are digital web addresses.", "hallucination"),
        ("Metadata is the actual data.", "Metadata describes properties of data.", "hallucination"),
        ("Machine learning prevents pattern recognition.", "ML algorithms identify and learn patterns.", "hallucination"),
        ("QR codes are barcodes only.", "QR codes are 2D codes, different from 1D barcodes.", "hallucination"),
        ("Hashtags delete social posts.", "Hashtags categorize and organize content.", "hallucination"),
        ("Spam filters send more junk.", "Spam filters block unwanted messages.", "hallucination"),
        ("Screenshots delete screen content.", "Screenshots capture and save screen images.", "hallucination"),
        ("Notifications disable all apps.", "Notifications alert users to events.", "hallucination"),
        ("Downloading uploads files.", "Downloading retrieves files from remote sources.", "hallucination"),
        ("Uploading deletes local files.", "Uploading sends copies while preserving originals.", "hallucination"),
        ("Hyperlinks break web pages.", "Hyperlinks connect and navigate between pages.", "hallucination"),
        ("Authentication reveals passwords.", "Authentication verifies identity securely.", "hallucination"),
        ("Syntax means random code.", "Syntax defines structured programming rules.", "hallucination"),

        # MATHEMATICS (80 pairs)
        # Factual
        ("A circle has no corners.", "Circles are round with no angular vertices.", "factual"),
        ("Prime numbers are divisible by one and themselves.", "Primes have exactly two factors.", "factual"),
        ("Parallel lines never intersect.", "Parallel lines maintain constant distance apart.", "factual"),
        ("Pi is approximately 3.14159.", "The mathematical constant pi represents circle ratios.", "factual"),
        ("Integers include negative numbers.", "Whole numbers span negative, zero, and positive values.", "factual"),
        ("A right angle measures 90 degrees.", "Perpendicular lines form 90-degree angles.", "factual"),
        ("Fractions represent parts of wholes.", "Fractional notation shows division of units.", "factual"),
        ("Even numbers are divisible by two.", "Numbers ending in 0, 2, 4, 6, 8 are even.", "factual"),
        ("The sum of angles in a triangle is 180°.", "Triangle interior angles total 180 degrees.", "factual"),
        ("Zero is neither positive nor negative.", "Zero is neutral on the number line.", "factual"),
        ("Multiplication is repeated addition.", "Multiplying combines equal groups.", "factual"),
        ("A square is a special rectangle.", "Squares have four equal sides and right angles.", "factual"),
        ("Percentages express parts per hundred.", "Percent means per one hundred.", "factual"),
        ("Exponents indicate repeated multiplication.", "Powers show how many times to multiply a base.", "factual"),
        ("The Pythagorean theorem applies to right triangles.", "a² + b² = c² for right triangles.", "factual"),
        ("Decimals represent fractional values.", "Decimal notation shows parts less than one.", "factual"),
        ("Probability ranges from 0 to 1.", "Likelihood is expressed between impossible and certain.", "factual"),
        ("Acute angles are less than 90 degrees.", "Sharp angles measure under 90 degrees.", "factual"),
        ("A pentagon has five sides.", "Pentagons are five-sided polygons.", "factual"),
        ("Negative times negative equals positive.", "Multiplying two negatives yields positive.", "factual"),
        ("Division is the inverse of multiplication.", "Division reverses multiplication operations.", "factual"),
        ("A radius is half a diameter.", "The radius extends from center to edge.", "factual"),
        ("Squares have four lines of symmetry.", "Squares can be folded four ways identically.", "factual"),
        ("The number 1 is not prime.", "Primes must have exactly two distinct factors.", "factual"),
        ("Obtuse angles exceed 90 degrees.", "Wide angles measure between 90 and 180 degrees.", "factual"),
        ("A hexagon has six sides.", "Hexagons are six-sided polygons.", "factual"),
        ("Mean is the arithmetic average.", "Mean sums values divided by count.", "factual"),
        ("Vertices are polygon corners.", "Vertices mark where sides meet.", "factual"),
        ("Circumference measures circle perimeter.", "Circle boundary length is circumference.", "factual"),
        ("Area measures surface space.", "Area quantifies two-dimensional extent.", "factual"),
        ("Volume measures three-dimensional space.", "Volume indicates capacity of 3D objects.", "factual"),
        ("Perpendicular lines form right angles.", "Perpendicular means at 90-degree intersection.", "factual"),
        ("Congruent shapes are identical.", "Congruent figures have same size and shape.", "factual"),
        ("A cube has six faces.", "Cubes are bounded by six square surfaces.", "factual"),
        ("Odd numbers are not divisible by two.", "Odd numbers leave remainder when divided by 2.", "factual"),
        ("A sphere is perfectly round.", "Spheres have uniform curvature in all directions.", "factual"),
        ("Median is the middle value.", "Median divides ordered data in half.", "factual"),
        ("Factors multiply to give a number.", "Factors are numbers that divide evenly.", "factual"),
        ("Absolute value is always non-negative.", "Absolute value represents distance from zero.", "factual"),
        ("A line extends infinitely.", "Lines have no endpoints in geometry.", "factual"),
        
        # Hallucination
        ("A circle has four corners.", "Circles are round without corners.", "hallucination"),
        ("Prime numbers are divisible by many numbers.", "Primes divide only by 1 and themselves.", "hallucination"),
        ("Parallel lines always intersect.", "Parallel lines never meet by definition.", "hallucination"),
        ("Pi equals exactly 3.", "Pi is approximately 3.14159, not exactly 3.", "hallucination"),
        ("Integers exclude negative numbers.", "Integers include negative, zero, and positive.", "hallucination"),
        ("A right angle measures 45 degrees.", "Right angles measure 90 degrees, not 45.", "hallucination"),
        ("Fractions represent whole numbers only.", "Fractions show parts of wholes.", "hallucination"),
        ("Even numbers are divisible by three.", "Even numbers are divisible by 2.", "hallucination"),
        ("Triangle angles sum to 360 degrees.", "Triangle angles total 180 degrees.", "hallucination"),
        ("Zero is a positive number.", "Zero is neither positive nor negative.", "hallucination"),
        ("Multiplication is repeated subtraction.", "Multiplication is repeated addition.", "hallucination"),
        ("A square has unequal sides.", "Squares have four equal sides.", "hallucination"),
        ("Percentages express parts per thousand.", "Percentages are parts per hundred.", "hallucination"),
        ("Exponents indicate addition.", "Exponents show repeated multiplication.", "hallucination"),
        ("Pythagorean theorem applies to all triangles.", "It applies only to right triangles.", "hallucination"),
        ("Decimals represent whole numbers only.", "Decimals can represent fractional parts.", "hallucination"),
        ("Probability ranges from -1 to 2.", "Probability ranges from 0 to 1.", "hallucination"),
        ("Acute angles exceed 90 degrees.", "Acute angles are less than 90 degrees.", "hallucination"),
        ("A pentagon has seven sides.", "Pentagons have five sides.", "hallucination"),
        ("Negative times negative equals negative.", "Negative times negative equals positive.", "hallucination"),
        ("Division is repeated addition.", "Division is inverse of multiplication.", "hallucination"),
        ("A radius equals the diameter.", "A radius is half the diameter.", "hallucination"),
        ("Squares have no symmetry.", "Squares have four lines of symmetry.", "hallucination"),
        ("The number 1 is prime.", "The number 1 is not considered prime.", "hallucination"),
        ("Obtuse angles are less than 45 degrees.", "Obtuse angles exceed 90 degrees.", "hallucination"),
        ("A hexagon has four sides.", "Hexagons have six sides.", "hallucination"),
        ("Mean is the most frequent value.", "Mean is the average; mode is most frequent.", "hallucination"),
        ("Vertices are polygon sides.", "Vertices are corners where sides meet.", "hallucination"),
        ("Circumference measures circle area.", "Circumference measures perimeter, not area.", "hallucination"),
        ("Area measures volume.", "Area measures 2D surface, volume measures 3D.", "hallucination"),
        ("Volume measures perimeter.", "Volume measures 3D space capacity.", "hallucination"),
        ("Perpendicular lines are parallel.", "Perpendicular lines intersect at right angles.", "hallucination"),
        ("Congruent shapes are different sizes.", "Congruent shapes are identical in size and shape.", "hallucination"),
        ("A cube has eight faces.", "A cube has six faces.", "hallucination"),
        ("Odd numbers are divisible by two.", "Odd numbers are not divisible evenly by 2.", "hallucination"),
        ("A sphere has flat sides.", "Spheres are perfectly round with no flat surfaces.", "hallucination"),
        ("Median is the average value.", "Median is the middle value when ordered.", "hallucination"),
        ("Factors add to give a number.", "Factors multiply to produce a number.", "hallucination"),
        ("Absolute value can be negative.", "Absolute value is always non-negative.", "hallucination"),
        ("A line has two endpoints.", "Lines extend infinitely without endpoints.", "hallucination"),

        # GEOGRAPHY & WORLD (90 pairs)
        # Factual
        ("The Equator divides hemispheres.", "The Equator separates north and south halves.", "factual"),
        ("Mountains form from tectonic activity.", "Plate movements create mountain ranges.", "factual"),
        ("Rivers flow toward lower elevations.", "Water naturally moves downhill to seas.", "factual"),
        ("Deserts receive little rainfall.", "Arid regions have minimal precipitation.", "factual"),
        ("Islands are surrounded by water.", "Land masses in oceans or lakes are islands.", "factual"),
        ("Continents are large landmasses.", "Earth has seven major continental divisions.", "factual"),
        ("Glaciers are moving ice masses.", "Large ice formations flow slowly over time.", "factual"),
        ("Oceans contain saltwater.", "Marine waters have dissolved salts.", "factual"),
        ("Latitude measures north-south position.", "Parallels indicate distance from equator.", "factual"),
        ("Longitude measures east-west position.", "Meridians show distance from prime meridian.", "factual"),
        ("Volcanoes erupt molten rock.", "Volcanic eruptions release lava and ash.", "factual"),
        ("Earthquakes result from tectonic shifts.", "Seismic activity occurs at fault lines.", "factual"),
        ("The Arctic is in the north.", "The Arctic region surrounds the North Pole.", "factual"),
        ("Antarctica is in the south.", "The Antarctic continent sits at the South Pole.", "factual"),
        ("Forests contain many trees.", "Wooded areas have dense tree populations.", "factual"),
        ("Valleys lie between mountains.", "Low areas form between higher elevations.", "factual"),
        ("Peninsulas are surrounded by water on three sides.", "Land projecting into water forms peninsulas.", "factual"),
        ("Seas are smaller than oceans.", "Seas are partially enclosed bodies of saltwater.", "factual"),
        ("Plateaus are elevated flat areas.", "High flat lands characterize plateau regions.", "factual"),
        ("Canyons are deep valleys.", "Erosion creates steep-sided gorges.", "factual"),
        ("Deltas form at river mouths.", "Sediment deposits create deltas where rivers meet seas.", "factual"),
        ("Monsoons are seasonal wind patterns.", "Monsoon systems bring periodic heavy rains.", "factual"),
        ("Tundra has frozen subsoil.", "Permafrost underlies tundra ecosystems.", "factual"),
        ("Coral reefs grow in warm waters.", "Tropical seas support coral formations.", "factual"),
        ("Icebergs are frozen freshwater.", "Glacial ice breaks off forming icebergs.", "factual"),
        ("Steppes are grassland plains.", "Temperate grasslands characterize steppe regions.", "factual"),
        ("Fjords are glacial valleys flooded by sea.", "Deep inlets form from glacial erosion.", "factual"),
        ("Archipelagos are island groups.", "Clusters of islands form archipelagos.", "factual"),
        ("Savanna has scattered trees.", "Tropical grasslands feature dispersed trees.", "factual"),
        ("Wetlands are saturated with water.", "Marshes and swamps have waterlogged soils.", "factual"),
        ("Trade winds blow toward the equator.", "Consistent tropical winds flow equatorward.", "factual"),
        ("The tropics are near the equator.", "Tropical zones span roughly 23°N to 23°S.", "factual"),
        ("Geysers erupt hot water.", "Thermal springs shoot water and steam periodically.", "factual"),
        ("Caves form underground.", "Natural cavities develop in rock formations.", "factual"),
        ("Sand dunes are wind-formed.", "Wind deposits create sandy hills.", "factual"),
        ("Atolls are ring-shaped coral islands.", "Circular reefs surround lagoons.", "factual"),
        ("Estuaries mix fresh and salt water.", "River mouths create brackish water zones.", "factual"),
        ("Taiga is boreal forest.", "Northern coniferous forests span high latitudes.", "factual"),
        ("Lagoons are shallow coastal waters.", "Protected water bodies form near shores.", "factual"),
        ("Cliffs are steep rock faces.", "Vertical or near-vertical slopes form cliffs.", "factual"),
        ("Mediterranean climate has dry summers.", "Warm dry seasons characterize this climate.", "factual"),
        ("Temperate zones have moderate climates.", "Mid-latitude regions experience seasonal variation.", "factual"),
        ("Polar regions have ice caps.", "Permanent ice covers polar areas.", "factual"),
        ("Tributaries feed main rivers.", "Smaller streams flow into larger rivers.", "factual"),
        ("Foothills lie below mountains.", "Lower elevations transition to mountain bases.", "factual"),
        
        # Hallucination
        ("The Equator runs through poles.", "The Equator circles Earth's middle, not poles.", "hallucination"),
        ("Mountains form from wind erosion.", "Mountains result from tectonic forces, not wind.", "hallucination"),
        ("Rivers flow uphill naturally.", "Rivers flow downhill due to gravity.", "hallucination"),
        ("Deserts receive constant rainfall.", "Deserts are defined by low precipitation.", "hallucination"),
        ("Islands are landlocked.", "Islands are surrounded by water by definition.", "hallucination"),
        ("Continents are small land areas.", "Continents are large landmasses.", "hallucination"),
        ("Glaciers are stationary rock.", "Glaciers are moving ice masses.", "hallucination"),
        ("Oceans contain freshwater.", "Oceans are saltwater bodies.", "hallucination"),
        ("Latitude measures east-west.", "Latitude measures north-south position.", "hallucination"),
        ("Longitude measures north-south.", "Longitude measures east-west position.", "hallucination"),
        ("Volcanoes erupt ice.", "Volcanoes erupt molten rock and ash.", "hallucination"),
        ("Earthquakes are caused by wind.", "Earthquakes result from tectonic movements.", "hallucination"),
        ("The Arctic is at the South Pole.", "The Arctic is at the North Pole.", "hallucination"),
        ("Antarctica is tropical.", "Antarctica is extremely cold polar region.", "hallucination"),
        ("Forests have no trees.", "Forests are defined by tree coverage.", "hallucination"),
        ("Valleys are higher than mountains.", "Valleys lie between mountains at lower elevations.", "hallucination"),
        ("Peninsulas are fully landlocked.", "Peninsulas are surrounded by water on three sides.", "hallucination"),
        ("Seas are larger than oceans.", "Oceans are larger than seas.", "hallucination"),
        ("Plateaus are deep valleys.", "Plateaus are elevated flat regions.", "hallucination"),
        ("Canyons are mountain peaks.", "Canyons are deep valleys, not peaks.", "hallucination"),
        ("Deltas form at mountain tops.", "Deltas form where rivers meet larger water bodies.", "hallucination"),
        ("Monsoons occur randomly.", "Monsoons are predictable seasonal patterns.", "hallucination"),
        ("Tundra has tropical heat.", "Tundra is characterized by cold temperatures.", "hallucination"),
        ("Coral reefs grow in ice.", "Coral reefs require warm tropical waters.", "hallucination"),
        ("Icebergs are saltwater.", "Icebergs are frozen freshwater from glaciers.", "hallucination"),
        ("Steppes are dense forests.", "Steppes are grassland plains, not forests.", "hallucination"),
        ("Fjords are desert formations.", "Fjords are coastal inlets formed by glaciers.", "hallucination"),
        ("Archipelagos are single islands.", "Archipelagos are groups of multiple islands.", "hallucination"),
        ("Savanna is densely forested.", "Savanna features grassland with scattered trees.", "hallucination"),
        ("Wetlands are completely dry.", "Wetlands are waterlogged by definition.", "hallucination"),
        ("Trade winds blow away from equator.", "Trade winds blow toward the equator.", "hallucination"),
        ("The tropics are at the poles.", "Tropics are regions near the equator.", "hallucination"),
        ("Geysers erupt cold water.", "Geysers erupt hot water and steam.", "hallucination"),
        ("Caves form above ground.", "Caves are underground cavities.", "hallucination"),
        ("Sand dunes form from rain.", "Sand dunes are shaped by wind.", "hallucination"),
        ("Atolls are mountain ranges.", "Atolls are ring-shaped coral reefs.", "hallucination"),
        ("Estuaries contain only freshwater.", "Estuaries mix fresh and salt water.", "hallucination"),
        ("Taiga is tropical rainforest.", "Taiga is northern coniferous forest.", "hallucination"),
        ("Lagoons are deep ocean trenches.", "Lagoons are shallow coastal waters.", "hallucination"),
        ("Cliffs are gentle slopes.", "Cliffs are steep vertical rock faces.", "hallucination"),
        ("Mediterranean has wet summers.", "Mediterranean climate features dry summers.", "hallucination"),
        ("Temperate zones are always frozen.", "Temperate zones have moderate seasonal climates.", "hallucination"),
        ("Polar regions are ice-free.", "Polar regions have permanent ice coverage.", "hallucination"),
        ("Tributaries flow from main rivers.", "Tributaries feed into main rivers.", "hallucination"),
        ("Foothills are taller than mountains.", "Foothills are lower elevations below mountains.", "hallucination"),

        # HISTORY & CULTURE (80 pairs)
        # Factual
        ("The Renaissance began in Italy.", "Italian city-states sparked the Renaissance movement.", "factual"),
        ("World War II ended in 1945.", "The Second World War concluded in 1945.", "factual"),
        ("The printing press revolutionized information.", "Gutenberg's invention enabled mass communication.", "factual"),
        ("Ancient Egypt built pyramids.", "Egyptians constructed monumental pyramid tombs.", "factual"),
        ("The Roman Empire was vast.", "Rome controlled extensive Mediterranean territories.", "factual"),
        ("Medieval castles provided defense.", "Fortified structures protected against attacks.", "factual"),
        ("The Industrial Revolution mechanized production.", "Manufacturing shifted from manual to machine work.", "factual"),
        ("Writing systems preserve knowledge.", "Written language records information across time.", "factual"),
        ("Democracy originated in ancient Greece.", "Athenians developed early democratic governance.", "factual"),
        ("The Cold War divided East and West.", "Ideological conflict separated global powers.", "factual"),
        ("Silk Road connected trade routes.", "Ancient pathways linked Asia and Europe commercially.", "factual"),
        ("The Enlightenment promoted reason.", "18th-century movement emphasized rational thought.", "factual"),
        ("Indigenous peoples have rich traditions.", "Native cultures maintain diverse heritage practices.", "factual"),
        ("Colonialism spread European influence.", "Imperial powers established overseas territories.", "factual"),
        ("The Bronze Age preceded the Iron Age.", "Metal technology progressed from bronze to iron.", "factual"),
        ("Hieroglyphics are ancient Egyptian writing.", "Pictorial symbols formed Egyptian script.", "factual"),
        ("The Magna Carta limited royal power.", "1215 charter established legal constraints.", "factual"),
        ("Vikings were Norse seafarers.", "Scandinavian explorers traveled extensively by sea.", "factual"),
        ("The Reformation challenged Catholic authority.", "Protestant movement questioned Church doctrines.", "factual"),
        ("Ancient China invented paper.", "Chinese developed paper-making techniques.", "factual"),
        ("The Ottoman Empire spanned centuries.", "Turkish empire lasted from 1299 to 1922.", "factual"),
        ("Philosophers question fundamental truths.", "Thinkers examine existence and knowledge.", "factual"),
        ("Mythology explains natural phenomena.", "Ancient stories interpret world origins.", "factual"),
        ("Feudalism structured medieval society.", "Hierarchical system organized land and labor.", "factual"),
        ("The Crusades were religious wars.", "Medieval conflicts focused on Holy Land control.", "factual"),
        ("Ancient Rome had aqueducts.", "Romans built water transport systems.", "factual"),
        ("The Maya developed a calendar.", "Mesoamerican civilization tracked astronomical cycles.", "factual"),
        ("Samurai were Japanese warriors.", "Elite fighters followed bushido code.", "factual"),
        ("The Aztecs ruled central Mexico.", "Powerful empire dominated Mesoamerican region.", "factual"),
        ("The Incas built Machu Picchu.", "Andean civilization constructed mountain citadel.", "factual"),
        ("Pharaohs ruled ancient Egypt.", "Egyptian kings held divine authority.", "factual"),
        ("Gladiators fought in Roman arenas.", "Trained combatants entertained crowds.", "factual"),
        ("The Black Death devastated Europe.", "Plague pandemic killed millions in 14th century.", "factual"),
        ("The American Revolution sought independence.", "Colonies fought for sovereignty from Britain.", "factual"),
        ("The French Revolution overthrew monarchy.", "1789 uprising ended royal absolutism.", "factual"),
        ("Nomadic tribes moved seasonally.", "Pastoral peoples migrated with herds.", "factual"),
        ("Ancient Greece had city-states.", "Independent poleis governed Greek territories.", "factual"),
        ("The Byzantine Empire preserved knowledge.", "Eastern Rome maintained classical learning.", "factual"),
        ("Persian Empire was powerful.", "Ancient Iran controlled vast territories.", "factual"),
        ("Cave paintings show early art.", "Prehistoric humans created rock imagery.", "factual"),
        
        # Hallucination
        ("The Renaissance began in Australia.", "The Renaissance started in Italy, not Australia.", "hallucination"),
        ("World War II ended in 1920.", "WWII ended in 1945, not 1920.", "hallucination"),
        ("The printing press was invented recently.", "Gutenberg invented the press in the 1440s.", "hallucination"),
        ("Ancient Egypt built skyscrapers.", "Egyptians built pyramids, not skyscrapers.", "hallucination"),
        ("The Roman Empire controlled Antarctica.", "Rome controlled Mediterranean, not Antarctica.", "hallucination"),
        ("Medieval castles were made of glass.", "Castles were built from stone, not glass.", "hallucination"),
        ("Industrial Revolution happened in ancient times.", "Industrial Revolution occurred 18th-19th century.", "hallucination"),
        ("Writing was invented last century.", "Writing systems emerged thousands of years ago.", "hallucination"),
        ("Democracy originated in China.", "Democracy originated in ancient Greece.", "hallucination"),
        ("The Cold War was a military battle.", "Cold War was ideological, not direct military conflict.", "hallucination"),
        ("Silk Road was an actual road.", "Silk Road was a network of trade routes.", "hallucination"),
        ("The Enlightenment promoted ignorance.", "The Enlightenment emphasized reason and knowledge.", "hallucination"),
        ("Indigenous peoples have no culture.", "Indigenous peoples have rich cultural traditions.", "hallucination"),
        ("Colonialism spread African influence.", "Colonialism spread European influence globally.", "hallucination"),
        ("The Iron Age preceded Bronze Age.", "Bronze Age came before Iron Age.", "hallucination"),
        ("Hieroglyphics are modern emojis.", "Hieroglyphics are ancient Egyptian symbols.", "hallucination"),
        ("Magna Carta expanded royal power.", "Magna Carta limited royal authority.", "hallucination"),
        ("Vikings were desert dwellers.", "Vikings were Norse seafarers.", "hallucination"),
        ("The Reformation supported Catholic Church.", "Reformation challenged Catholic authority.", "hallucination"),
        ("Ancient Rome invented paper.", "Ancient China invented paper.", "hallucination"),
        ("Ottoman Empire lasted 50 years.", "Ottoman Empire spanned over 600 years.", "hallucination"),
        ("Philosophers avoid questioning.", "Philosophers fundamentally question truths.", "hallucination"),
        ("Mythology is modern science.", "Mythology consists of ancient explanatory stories.", "hallucination"),
        ("Feudalism was democratic.", "Feudalism was hierarchical, not democratic.", "hallucination"),
        ("The Crusades were about trade.", "Crusades were primarily religious conflicts.", "hallucination"),
        ("Ancient Rome had no water systems.", "Romans built extensive aqueduct systems.", "hallucination"),
        ("The Maya had no calendar.", "Maya developed sophisticated calendar systems.", "hallucination"),
        ("Samurai were European knights.", "Samurai were Japanese warriors.", "hallucination"),
        ("The Aztecs ruled South America.", "Aztecs ruled central Mexico, not South America.", "hallucination"),
        ("The Incas built the Colosseum.", "Romans built the Colosseum; Incas built Machu Picchu.", "hallucination"),
        ("Pharaohs ruled ancient China.", "Pharaohs ruled ancient Egypt.", "hallucination"),
        ("Gladiators were peaceful monks.", "Gladiators were combat fighters.", "hallucination"),
        ("Black Death was a minor illness.", "Black Death killed millions across Europe.", "hallucination"),
        ("American Revolution sought British rule.", "Revolution sought independence from Britain.", "hallucination"),
        ("French Revolution strengthened monarchy.", "French Revolution overthrew the monarchy.", "hallucination"),
        ("Nomadic tribes stayed in one place.", "Nomadic tribes moved seasonally by definition.", "hallucination"),
        ("Ancient Greece was united.", "Greece consisted of independent city-states.", "hallucination"),
        ("Byzantine Empire destroyed knowledge.", "Byzantine Empire preserved classical knowledge.", "hallucination"),
        ("Persian Empire was weak.", "Persian Empire was powerful and extensive.", "hallucination"),
        ("Cave paintings are modern graffiti.", "Cave paintings are prehistoric art.", "hallucination"),

        # BIOLOGY & LIFE SCIENCES (90 pairs)
        # Factual
        ("Cells are the basic unit of life.", "All living organisms consist of cells.", "factual"),
        ("Mitochondria produce cellular energy.", "Mitochondria generate ATP for cells.", "factual"),
        ("Chloroplasts enable photosynthesis.", "Plant cells use chloroplasts to capture light.", "factual"),
        ("DNA stores genetic information.", "Deoxyribonucleic acid contains hereditary data.", "factual"),
        ("Proteins are made of amino acids.", "Amino acid chains form protein structures.", "factual"),
        ("Enzymes catalyze chemical reactions.", "Biological catalysts speed up processes.", "factual"),
        ("Antibodies fight infections.", "Immune proteins target foreign invaders.", "factual"),
        ("Neurons transmit electrical signals.", "Nerve cells communicate via impulses.", "factual"),
        ("Red blood cells carry oxygen.", "Erythrocytes transport oxygen through blood.", "factual"),
        ("The nucleus contains chromosomes.", "Genetic material is housed in nucleus.", "factual"),
        ("Metamorphosis transforms organisms.", "Animals undergo developmental stage changes.", "factual"),
        ("Ecosystems include living and nonliving components.", "Biotic and abiotic factors interact.", "factual"),
        ("Food chains show energy flow.", "Organisms transfer energy through consumption.", "factual"),
        ("Symbiosis involves species interaction.", "Different organisms live in close relationships.", "factual"),
        ("Evolution occurs through natural selection.", "Favorable traits increase in populations.", "factual"),
        ("Genes determine inherited traits.", "Genetic code influences characteristics.", "factual"),
        ("Bacteria are single-celled prokaryotes.", "Bacterial cells lack membrane-bound nuclei.", "factual"),
        ("Viruses require host cells.", "Viral particles need cells to reproduce.", "factual"),
        ("Fungi decompose organic matter.", "Fungal organisms break down dead material.", "factual"),
        ("Ribosomes synthesize proteins.", "Cellular structures assemble amino acids.", "factual"),
        ("Meiosis produces reproductive cells.", "Cell division creates gametes with half chromosomes.", "factual"),
        ("Homeostasis maintains internal balance.", "Organisms regulate stable internal conditions.", "factual"),
        ("Hormones are chemical messengers.", "Endocrine signals regulate body functions.", "factual"),
        ("Photosynthesis converts light to chemical energy.", "Plants transform sunlight into glucose.", "factual"),
        ("Respiration releases energy from glucose.", "Cells break down sugar for ATP.", "factual"),
        ("Adaptation increases survival.", "Beneficial traits improve fitness.", "factual"),
        ("Biodiversity measures variety of life.", "Ecosystem health includes species diversity.", "factual"),
        ("Predators hunt other animals.", "Carnivores consume prey for sustenance.", "factual"),
        ("Herbivores eat plant matter.", "Plant-eating animals are herbivorous.", "factual"),
        ("Omnivores consume both plants and animals.", "Varied diet includes multiple food sources.", "factual"),
        ("Parasites benefit at host expense.", "Parasitic organisms harm their hosts.", "factual"),
        ("Mutations alter DNA sequences.", "Genetic changes modify hereditary code.", "factual"),
        ("Species share common ancestry.", "Related organisms descended from ancestors.", "factual"),
        ("Tissues group similar cells.", "Cell collections form functional tissues.", "factual"),
        ("Organs perform specific functions.", "Tissue combinations create organs.", "factual"),
        ("Systems coordinate organ activities.", "Multiple organs work together systematically.", "factual"),
        ("Camouflage provides protection.", "Concealment helps organisms avoid detection.", "factual"),
        ("Migration follows seasonal patterns.", "Animals travel periodically to different regions.", "factual"),
        ("Hibernation conserves energy.", "Dormancy reduces metabolic needs.", "factual"),
        ("Pollination enables plant reproduction.", "Pollen transfer facilitates fertilization.", "factual"),
        ("Seeds contain embryonic plants.", "Plant embryos are packaged in seeds.", "factual"),
        ("Spores produce new organisms.", "Reproductive cells develop into individuals.", "factual"),
        ("Fossils preserve ancient life.", "Mineralized remains show past organisms.", "factual"),
        ("Extinction eliminates species.", "Species permanently disappear from Earth.", "factual"),
        ("Biomes have characteristic climates.", "Major ecosystems feature distinct conditions.", "factual"),
        
        # Hallucination
        ("Cells are the largest life unit.", "Cells are the smallest basic unit of life.", "hallucination"),
        ("Mitochondria store genetic information.", "Mitochondria produce energy, DNA stores genetics.", "hallucination"),
        ("Chloroplasts are in animal cells.", "Chloroplasts are found in plant cells only.", "hallucination"),
        ("DNA is made of proteins.", "DNA is made of nucleotides, not proteins.", "hallucination"),
        ("Proteins are made of lipids.", "Proteins are composed of amino acids.", "hallucination"),
        ("Enzymes slow down reactions.", "Enzymes speed up chemical reactions.", "hallucination"),
        ("Antibodies cause infections.", "Antibodies fight infections, not cause them.", "hallucination"),
        ("Neurons carry blood.", "Neurons transmit electrical signals, not blood.", "hallucination"),
        ("Red blood cells carry nutrients only.", "Red blood cells primarily carry oxygen.", "hallucination"),
        ("The nucleus produces energy.", "Mitochondria produce energy; nucleus stores DNA.", "hallucination"),
        ("Metamorphosis is instantaneous.", "Metamorphosis is a gradual transformation process.", "hallucination"),
        ("Ecosystems include only living things.", "Ecosystems include both biotic and abiotic factors.", "hallucination"),
        ("Food chains show water flow.", "Food chains show energy flow through consumption.", "hallucination"),
        ("Symbiosis means competition only.", "Symbiosis involves various interactive relationships.", "hallucination"),
        ("Evolution occurs instantly.", "Evolution happens gradually over generations.", "hallucination"),
        ("Environment determines all traits.", "Genes and environment both influence traits.", "hallucination"),
        ("Bacteria have complex nuclei.", "Bacteria are prokaryotes without true nuclei.", "hallucination"),
        ("Viruses are independent organisms.", "Viruses require host cells to reproduce.", "hallucination"),
        ("Fungi are plants.", "Fungi are a separate kingdom from plants.", "hallucination"),
        ("Ribosomes store DNA.", "Ribosomes synthesize proteins, not store DNA.", "hallucination"),
        ("Meiosis creates identical cells.", "Meiosis produces varied reproductive cells.", "hallucination"),
        ("Homeostasis causes imbalance.", "Homeostasis maintains internal balance.", "hallucination"),
        ("Hormones are physical structures.", "Hormones are chemical messengers.", "hallucination"),
        ("Photosynthesis occurs in animals.", "Photosynthesis occurs in plants and some bacteria.", "hallucination"),
        ("Respiration requires light.", "Respiration occurs with or without light.", "hallucination"),
        ("Adaptation harms survival.", "Adaptation increases survival chances.", "hallucination"),
        ("Biodiversity means single species.", "Biodiversity indicates variety of species.", "hallucination"),
        ("Predators eat only plants.", "Predators hunt and consume other animals.", "hallucination"),
        ("Herbivores eat meat.", "Herbivores consume plant matter only.", "hallucination"),
        ("Omnivores eat rocks.", "Omnivores eat both plants and animals.", "hallucination"),
        ("Parasites help their hosts.", "Parasites harm hosts for their benefit.", "hallucination"),
        ("Mutations never change DNA.", "Mutations alter DNA sequences by definition.", "hallucination"),
        ("Species have no common ancestry.", "Species share evolutionary common ancestors.", "hallucination"),
        ("Tissues are individual cells.", "Tissues are groups of similar cells.", "hallucination"),
        ("Organs are single cells.", "Organs consist of multiple tissue types.", "hallucination"),
        ("Systems are individual organs.", "Systems coordinate multiple organs.", "hallucination"),
        ("Camouflage attracts predators.", "Camouflage helps avoid predator detection.", "hallucination"),
        ("Migration is random movement.", "Migration follows predictable seasonal patterns.", "hallucination"),
        ("Hibernation increases activity.", "Hibernation reduces activity to conserve energy.", "hallucination"),
        ("Pollination prevents reproduction.", "Pollination enables plant reproduction.", "hallucination"),
        ("Seeds contain fully grown plants.", "Seeds contain plant embryos, not mature plants.", "hallucination"),
        ("Spores are dead cells.", "Spores are living reproductive cells.", "hallucination"),
        ("Fossils show current life only.", "Fossils preserve evidence of ancient life.", "hallucination"),
        ("Extinction creates new species.", "Extinction eliminates species permanently.", "hallucination"),
        ("Biomes have identical climates.", "Biomes have characteristic distinct climates.", "hallucination"),

        # CHEMISTRY (70 pairs)
        # Factual
        ("Atoms are composed of subatomic particles.", "Protons, neutrons, and electrons form atoms.", "factual"),
        ("Elements cannot be broken down chemically.", "Pure substances consist of one atom type.", "factual"),
        ("Compounds contain multiple elements.", "Chemical bonds join different atoms.", "factual"),
        ("Chemical reactions rearrange atoms.", "Bonds break and form during reactions.", "factual"),
        ("Acids have low pH values.", "Acidic solutions measure below pH 7.", "factual"),
        ("Bases have high pH values.", "Basic solutions measure above pH 7.", "factual"),
        ("Catalysts speed reactions without being consumed.", "Substances accelerate reactions while remaining unchanged.", "factual"),
        ("Oxidation involves electron loss.", "Atoms lose electrons during oxidation.", "factual"),
        ("Reduction involves electron gain.", "Atoms gain electrons during reduction.", "factual"),
        ("The periodic table organizes elements.", "Elements are arranged by atomic properties.", "factual"),
        ("Noble gases are unreactive.", "Group 18 elements rarely form compounds.", "factual"),
        ("Covalent bonds share electrons.", "Atoms share electron pairs in bonds.", "factual"),
        ("Ionic bonds transfer electrons.", "Electrons move from one atom to another.", "factual"),
        ("Molecules are bonded atoms.", "Chemical structures contain multiple atoms.", "factual"),
        ("Solutions are homogeneous mixtures.", "Dissolved substances distribute uniformly.", "factual"),
        ("Solubility measures dissolving capacity.", "Maximum amount dissolved at conditions.", "factual"),
        ("Endothermic reactions absorb energy.", "Heat is taken in during reactions.", "factual"),
        ("Exothermic reactions release energy.", "Heat is given off during reactions.", "factual"),
        ("Isotopes have different neutron numbers.", "Same element with varied neutron counts.", "factual"),
        ("Atomic number indicates proton count.", "Element identity determined by protons.", "factual"),
        ("Atomic mass includes protons and neutrons.", "Nuclear particles contribute to mass.", "factual"),
        ("Electrons occupy energy levels.", "Electron shells surround atomic nuclei.", "factual"),
        ("Valence electrons determine reactivity.", "Outer shell electrons participate in bonding.", "factual"),
        ("Metals conduct electricity.", "Metallic elements allow current flow.", "factual"),
        ("Nonmetals are poor conductors.", "Non-metallic elements resist current.", "factual"),
        ("Alloys combine multiple metals.", "Metal mixtures create new materials.", "factual"),
        ("Polymers are large chain molecules.", "Repeating units form long structures.", "factual"),
        ("Organic compounds contain carbon.", "Carbon-based molecules are organic.", "factual"),
        ("Inorganic compounds lack carbon-hydrogen bonds.", "Non-organic substances differ structurally.", "factual"),
        ("Crystalline solids have ordered structure.", "Atoms arrange in regular patterns.", "factual"),
        ("Amorphous solids lack order.", "Random atomic arrangements characterize them.", "factual"),
        ("Phase changes alter physical state.", "Matter transitions between solid, liquid, gas.", "factual"),
        ("Sublimation transitions solid to gas.", "Direct conversion without liquid phase.", "factual"),
        ("Deposition transitions gas to solid.", "Direct conversion without liquid phase.", "factual"),
        ("Concentration measures solute amount.", "Quantity of dissolved substance per volume.", "factual"),
        
        # Hallucination
        ("Atoms cannot be divided.", "Atoms consist of smaller subatomic particles.", "hallucination"),
        ("Elements can be chemically broken down.", "Elements are pure substances that cannot decompose.", "hallucination"),
        ("Compounds contain one element only.", "Compounds consist of multiple bonded elements.", "hallucination"),
        ("Chemical reactions create atoms.", "Reactions rearrange existing atoms.", "hallucination"),
        ("Acids have high pH values.", "Acids have low pH below 7.", "hallucination"),
        ("Bases have low pH values.", "Bases have high pH above 7.", "hallucination"),
        ("Catalysts are consumed in reactions.", "Catalysts remain unchanged after reactions.", "hallucination"),
        ("Oxidation involves electron gain.", "Oxidation involves electron loss.", "hallucination"),
        ("Reduction involves electron loss.", "Reduction involves electron gain.", "hallucination"),
        ("The periodic table is random.", "Periodic table organizes elements systematically.", "hallucination"),
        ("Noble gases are highly reactive.", "Noble gases are chemically unreactive.", "hallucination"),
        ("Covalent bonds transfer electrons.", "Covalent bonds share electrons between atoms.", "hallucination"),
        ("Ionic bonds share electrons.", "Ionic bonds involve electron transfer.", "hallucination"),
        ("Molecules are single atoms.", "Molecules contain two or more bonded atoms.", "hallucination"),
        ("Solutions are heterogeneous.", "Solutions are homogeneous mixtures.", "hallucination"),
        ("Solubility measures temperature.", "Solubility measures dissolving capacity.", "hallucination"),
        ("Endothermic reactions release energy.", "Endothermic reactions absorb energy.", "hallucination"),
        ("Exothermic reactions absorb energy.", "Exothermic reactions release energy.", "hallucination"),
        ("Isotopes have different proton numbers.", "Isotopes differ in neutron count, not protons.", "hallucination"),
        ("Atomic number indicates neutron count.", "Atomic number indicates proton count.", "hallucination"),
        ("Atomic mass includes only protons.", "Atomic mass includes protons and neutrons.", "hallucination"),
        ("Electrons are in the nucleus.", "Electrons orbit the nucleus in shells.", "hallucination"),
        ("Inner electrons determine reactivity.", "Valence outer electrons determine reactivity.", "hallucination"),
        ("Metals don't conduct electricity.", "Metals are excellent electrical conductors.", "hallucination"),
        ("Nonmetals conduct electricity well.", "Nonmetals are poor electrical conductors.", "hallucination"),
        ("Alloys are pure metals.", "Alloys are mixtures of multiple metals.", "hallucination"),
        ("Polymers are small molecules.", "Polymers are large chain molecules.", "hallucination"),
        ("Organic compounds lack carbon.", "Organic compounds contain carbon by definition.", "hallucination"),
        ("Inorganic compounds contain carbon-hydrogen.", "Inorganic generally lack C-H bonds.", "hallucination"),
        ("Crystalline solids are random.", "Crystalline solids have ordered structures.", "hallucination"),
        ("Amorphous solids are ordered.", "Amorphous solids lack regular structure.", "hallucination"),
        ("Phase changes alter chemical composition.", "Phase changes alter state, not composition.", "hallucination"),
        ("Sublimation goes through liquid phase.", "Sublimation skips liquid phase directly.", "hallucination"),
        ("Deposition requires liquid phase.", "Deposition transitions gas to solid directly.", "hallucination"),
        ("Concentration measures temperature.", "Concentration measures solute amount.", "hallucination"),

        # PHYSICS (80 pairs)
        # Factual
        ("Gravity attracts objects with mass.", "Massive bodies exert gravitational force.", "factual"),
        ("Speed measures distance per time.", "Velocity indicates how fast objects move.", "factual"),
        ("Acceleration is change in velocity.", "Objects speed up or slow down.", "factual"),
        ("Force equals mass times acceleration.", "Newton's second law describes motion.", "factual"),
        ("Energy cannot be created or destroyed.", "Conservation law preserves total energy.", "factual"),
        ("Kinetic energy involves motion.", "Moving objects possess kinetic energy.", "factual"),
        ("Potential energy is stored energy.", "Position or configuration stores energy.", "factual"),
        ("Work transfers energy.", "Force applied over distance does work.", "factual"),
        ("Power measures work rate.", "Energy transfer speed defines power.", "factual"),
        ("Momentum equals mass times velocity.", "Moving objects have momentum.", "factual"),
        ("Friction opposes motion.", "Resistance force acts against movement.", "factual"),
        ("Inertia resists motion changes.", "Objects maintain their state of motion.", "factual"),
        ("Waves transfer energy without matter.", "Disturbances propagate through media.", "factual"),
        ("Frequency measures wave cycles.", "Oscillations per unit time define frequency.", "factual"),
        ("Wavelength is distance between crests.", "Spatial period of wave pattern.", "factual"),
        ("Amplitude indicates wave intensity.", "Height of wave relates to energy.", "factual"),
        ("Light travels in waves.", "Electromagnetic radiation propagates.", "factual"),
        ("Reflection bounces light off surfaces.", "Light rays return from interfaces.", "factual"),
        ("Refraction bends light through media.", "Light changes direction in materials.", "factual"),
        ("Mirrors reflect light.", "Reflective surfaces return light rays.", "factual"),
        ("Lenses focus or spread light.", "Curved glass bends light paths.", "factual"),
        ("Convex lenses converge light.", "Outward-curved lenses bring rays together.", "factual"),
        ("Concave lenses diverge light.", "Inward-curved lenses spread rays apart.", "factual"),
        ("Electricity flows through conductors.", "Charged particles move in materials.", "factual"),
        ("Current measures charge flow rate.", "Electrical current indicates electron movement.", "factual"),
        ("Voltage is electrical potential difference.", "Energy per charge defines voltage.", "factual"),
        ("Resistance opposes current flow.", "Materials impede electron movement.", "factual"),
        ("Circuits provide current paths.", "Closed loops allow electricity flow.", "factual"),
        ("Magnets have north and south poles.", "Magnetic dipoles attract and repel.", "factual"),
        ("Magnetic fields surround magnets.", "Invisible force fields extend from poles.", "factual"),
        ("Electric and magnetic fields relate.", "Electromagnetic forces are interconnected.", "factual"),
        ("Temperature measures thermal energy.", "Heat relates to particle motion.", "factual"),
        ("Heat flows from hot to cold.", "Thermal energy transfers to lower temperature.", "factual"),
        ("Conduction transfers heat through contact.", "Direct touch conducts thermal energy.", "factual"),
        ("Convection transfers heat through fluids.", "Fluid motion carries thermal energy.", "factual"),
        ("Radiation transfers heat through waves.", "Electromagnetic waves carry thermal energy.", "factual"),
        ("Pressure is force per area.", "Distributed force defines pressure.", "factual"),
        ("Density is mass per volume.", "Compactness of matter defines density.", "factual"),
        ("Buoyancy makes objects float.", "Upward force counteracts weight.", "factual"),
        ("Sound requires a medium.", "Mechanical waves need matter to travel.", "factual"),
        
        # Hallucination
        ("Gravity repels objects with mass.", "Gravity attracts, not repels masses.", "hallucination"),
        ("Speed measures temperature.", "Speed measures distance per time.", "hallucination"),
        ("Acceleration is constant velocity.", "Acceleration is change in velocity.", "hallucination"),
        ("Force equals mass divided by time.", "Force equals mass times acceleration.", "hallucination"),
        ("Energy can be created freely.", "Energy is conserved, not created or destroyed.", "hallucination"),
        ("Kinetic energy involves stationary objects.", "Kinetic energy requires motion.", "hallucination"),
        ("Potential energy is motion energy.", "Potential energy is stored, not motion energy.", "hallucination"),
        ("Work requires no force.", "Work requires force applied over distance.", "hallucination"),
        ("Power measures distance.", "Power measures rate of doing work.", "hallucination"),
        ("Momentum equals mass divided by velocity.", "Momentum equals mass times velocity.", "hallucination"),
        ("Friction assists motion.", "Friction opposes motion.", "hallucination"),
        ("Inertia creates motion.", "Inertia resists changes in motion.", "hallucination"),
        ("Waves transfer matter.", "Waves transfer energy, not matter.", "hallucination"),
        ("Frequency measures wave height.", "Frequency measures cycles per time unit.", "hallucination"),
        ("Wavelength is wave height.", "Wavelength is distance between crests.", "hallucination"),
        ("Amplitude indicates wave speed.", "Amplitude indicates wave intensity or energy.", "hallucination"),
        ("Light travels as particles only.", "Light exhibits both wave and particle properties.", "hallucination"),
        ("Reflection absorbs all light.", "Reflection bounces light off surfaces.", "hallucination"),
        ("Refraction stops light completely.", "Refraction bends light through media.", "hallucination"),
        ("Mirrors absorb all light.", "Mirrors reflect light.", "hallucination"),
        ("Lenses block all light.", "Lenses focus or spread light.", "hallucination"),
        ("Convex lenses spread light.", "Convex lenses converge light rays.", "hallucination"),
        ("Concave lenses focus light.", "Concave lenses diverge light rays.", "hallucination"),
        ("Electricity cannot flow.", "Electricity flows through conductors.", "hallucination"),
        ("Current measures voltage.", "Current measures charge flow rate.", "hallucination"),
        ("Voltage is electrical resistance.", "Voltage is potential difference, not resistance.", "hallucination"),
        ("Resistance helps current flow.", "Resistance opposes current flow.", "hallucination"),
        ("Circuits block electricity.", "Circuits provide paths for current.", "hallucination"),
        ("Magnets have three poles.", "Magnets have two poles: north and south.", "hallucination"),
        ("Magnetic fields are visible.", "Magnetic fields are invisible forces.", "hallucination"),
        ("Electric and magnetic fields are unrelated.", "Electromagnetic fields are interconnected.", "hallucination"),
        ("Temperature measures distance.", "Temperature measures thermal energy.", "hallucination"),
        ("Heat flows from cold to hot.", "Heat flows from hot to cold regions.", "hallucination"),
        ("Conduction requires no contact.", "Conduction transfers heat through contact.", "hallucination"),
        ("Convection occurs in solids.", "Convection transfers heat through fluids.", "hallucination"),
        ("Radiation requires matter.", "Radiation transfers heat through vacuum.", "hallucination"),
        ("Pressure is force times area.", "Pressure is force per unit area.", "hallucination"),
        ("Density is mass times volume.", "Density is mass per unit volume.", "hallucination"),
        ("Buoyancy makes objects sink always.", "Buoyancy can make objects float.", "hallucination"),
        ("Sound travels through vacuum.", "Sound requires a medium to propagate.", "hallucination"),

        # ASTRONOMY & SPACE (60 pairs)
        # Factual
        ("Stars produce light and heat.", "Nuclear fusion powers stellar radiation.", "factual"),
        ("Planets orbit stars.", "Gravitational attraction keeps planets in orbit.", "factual"),
        ("Moons orbit planets.", "Natural satellites circle their parent planets.", "factual"),
        ("The solar system has eight planets.", "Our system contains Mercury through Neptune.", "factual"),
        ("The Sun is a medium-sized star.", "Our star is classified as a yellow dwarf.", "factual"),
        ("Black holes have extreme gravity.", "Collapsed stars create regions of immense gravitational pull.", "factual"),
        ("Galaxies contain billions of stars.", "Star systems cluster in vast cosmic structures.", "factual"),
        ("The Milky Way is our galaxy.", "Our solar system resides in this spiral galaxy.", "factual"),
        ("Light years measure cosmic distances.", "Distance light travels in one year.", "factual"),
        ("The universe is expanding.", "Space itself stretches over time.", "factual"),
        ("Comets have icy compositions.", "Frozen bodies develop tails near Sun.", "factual"),
        ("Asteroids are rocky objects.", "Small celestial bodies orbit the Sun.", "factual"),
        ("Meteors burn in atmosphere.", "Space rocks create shooting stars.", "factual"),
        ("Meteorites reach Earth's surface.", "Surviving space rocks impact ground.", "factual"),
        ("Constellations are star patterns.", "Apparent groupings form recognizable shapes.", "factual"),
        ("The Big Bang began the universe.", "Cosmic expansion started from singularity.", "factual"),
        ("Red giants are expanded stars.", "Aging stars swell to enormous sizes.", "factual"),
        ("White dwarfs are stellar remnants.", "Collapsed cores remain after fusion ends.", "factual"),
        ("Supernovae are stellar explosions.", "Massive stars explode catastrophically.", "factual"),
        ("Nebulae are interstellar clouds.", "Gas and dust form cosmic structures.", "factual"),
        ("Solar eclipses block sunlight.", "Moon passes between Earth and Sun.", "factual"),
        ("Lunar eclipses darken the Moon.", "Earth's shadow falls on lunar surface.", "factual"),
        ("Tides result from lunar gravity.", "Moon's pull creates ocean movements.", "factual"),
        ("Space is mostly vacuum.", "Near-empty regions separate celestial bodies.", "factual"),
        ("Gravity decreases with distance.", "Gravitational force weakens farther from mass.", "factual"),
        ("Orbits follow elliptical paths.", "Planetary motion traces oval trajectories.", "factual"),
        ("Saturn has prominent rings.", "Ice and rock particles circle the planet.", "factual"),
        ("Jupiter is the largest planet.", "Gas giant exceeds all other planets in size.", "factual"),
        ("Mars appears red.", "Iron oxide covers Martian surface.", "factual"),
        ("Venus has a thick atmosphere.", "Dense carbon dioxide surrounds the planet.", "factual"),
        
        # Hallucination
        ("Stars are cold objects.", "Stars produce tremendous heat and light.", "hallucination"),
        ("Planets orbit moons.", "Planets orbit stars, not moons.", "hallucination"),
        ("Moons orbit stars directly.", "Moons orbit planets, not stars directly.", "hallucination"),
        ("The solar system has twelve planets.", "Our solar system has eight recognized planets.", "hallucination"),
        ("The Sun is the largest star.", "The Sun is medium-sized among stars.", "hallucination"),
        ("Black holes emit light.", "Black holes trap light with extreme gravity.", "hallucination"),
        ("Galaxies contain ten stars.", "Galaxies contain billions of stars.", "hallucination"),
        ("The Milky Way is a planet.", "The Milky Way is our galaxy.", "hallucination"),
        ("Light years measure time only.", "Light years measure distance in space.", "hallucination"),
        ("The universe is shrinking.", "The universe is expanding continuously.", "hallucination"),
        ("Comets are pure metal.", "Comets have icy and rocky compositions.", "hallucination"),
        ("Asteroids are gaseous clouds.", "Asteroids are solid rocky objects.", "hallucination"),
        ("Meteors originate on Earth.", "Meteors are space rocks entering atmosphere.", "hallucination"),
        ("Meteorites never reach ground.", "Meteorites are space rocks that impact surface.", "hallucination"),
        ("Constellations are physical groups.", "Constellations are apparent patterns from Earth.", "hallucination"),
        ("The Big Bang was recent.", "Big Bang occurred billions of years ago.", "hallucination"),
        ("Red giants are small stars.", "Red giants are enormously expanded stars.", "hallucination"),
        ("White dwarfs are growing.", "White dwarfs are collapsed stellar remnants.", "hallucination"),
        ("Supernovae create stars.", "Supernovae are stellar explosions, not births.", "hallucination"),
        ("Nebulae are solid objects.", "Nebulae are clouds of gas and dust.", "hallucination"),
        ("Solar eclipses occur at night.", "Solar eclipses occur during daytime.", "hallucination"),
        ("Lunar eclipses brighten the Moon.", "Lunar eclipses darken the Moon.", "hallucination"),
        ("Tides result from wind.", "Tides result primarily from lunar gravity.", "hallucination"),
        ("Space is filled with air.", "Space is mostly vacuum with sparse matter.", "hallucination"),
        ("Gravity increases with distance.", "Gravity decreases with increasing distance.", "hallucination"),
        ("Orbits are perfect circles.", "Orbits are typically elliptical, not circular.", "hallucination"),
        ("Saturn has no rings.", "Saturn is famous for its ring system.", "hallucination"),
        ("Jupiter is the smallest planet.", "Jupiter is the largest planet in our system.", "hallucination"),
        ("Mars appears blue.", "Mars appears red due to iron oxide.", "hallucination"),
        ("Venus has no atmosphere.", "Venus has an extremely thick atmosphere.", "hallucination"),

        # HEALTH & MEDICINE (50 pairs)
        # Factual
        ("Vaccines prevent diseases.", "Immunizations train immune systems.", "factual"),
        ("Antibiotics kill bacteria.", "Medicines target bacterial infections.", "factual"),
        ("Vitamins are essential nutrients.", "Organic compounds support body functions.", "factual"),
        ("Exercise strengthens muscles.", "Physical activity builds muscle tissue.", "factual"),
        ("Sleep is essential for health.", "Rest allows body recovery and repair.", "factual"),
        ("Hydration maintains body functions.", "Water supports cellular processes.", "factual"),
        ("Balanced diet provides nutrition.", "Varied food intake supplies necessary nutrients.", "factual"),
        ("Stress affects mental health.", "Psychological pressure impacts wellbeing.", "factual"),
        ("Smoking damages lungs.", "Tobacco use harms respiratory system.", "factual"),
        ("Alcohol affects the liver.", "Excessive drinking damages hepatic tissue.", "factual"),
        ("Diabetes affects blood sugar.", "Condition impairs glucose regulation.", "factual"),
        ("Blood pressure indicates cardiovascular health.", "Arterial force reflects heart function.", "factual"),
        ("Cholesterol affects heart disease risk.", "Lipid levels influence cardiac health.", "factual"),
        ("Cancer involves abnormal cell growth.", "Uncontrolled division characterizes malignancy.", "factual"),
        ("Inflammation is immune response.", "Body reacts to injury or infection.", "factual"),
        ("Pain signals tissue damage.", "Discomfort alerts to potential harm.", "factual"),
        ("Fever indicates infection.", "Elevated temperature fights pathogens.", "factual"),
        ("Allergies are immune overreactions.", "Hypersensitivity to harmless substances.", "factual"),
        ("Asthma affects breathing.", "Airways narrow causing respiratory difficulty.", "factual"),
        ("Osteoporosis weakens bones.", "Decreased bone density increases fracture risk.", "factual"),
        ("Arthritis causes joint pain.", "Inflammation affects joint mobility.", "factual"),
        ("Alzheimer's affects memory.", "Neurodegenerative disease impairs cognition.", "factual"),
        ("Depression impacts mood.", "Mental condition causes persistent sadness.", "factual"),
        ("Anxiety causes worry.", "Excessive fear characterizes condition.", "factual"),
        ("Hygiene prevents disease spread.", "Cleanliness reduces pathogen transmission.", "factual"),
        ("Quarantine limits infection spread.", "Isolation prevents disease transmission.", "factual"),
        ("Pandemics are global disease outbreaks.", "Widespread epidemics cross borders.", "factual"),
        
        # Hallucination
        ("Vaccines cause diseases.", "Vaccines prevent diseases by training immunity.", "hallucination"),
        ("Antibiotics kill viruses.", "Antibiotics target bacteria, not viruses.", "hallucination"),
        ("Vitamins are harmful.", "Vitamins are essential beneficial nutrients.", "hallucination"),
        ("Exercise weakens muscles.", "Exercise strengthens and builds muscles.", "hallucination"),
        ("Sleep is unnecessary.", "Sleep is essential for health and recovery.", "hallucination"),
        ("Hydration is harmful.", "Hydration is necessary for body functions.", "hallucination"),
        ("Junk food is most nutritious.", "Balanced diet provides better nutrition.", "hallucination"),
        ("Stress improves mental health.", "Stress negatively affects mental health.", "hallucination"),
        ("Smoking improves lung function.", "Smoking severely damages lungs.", "hallucination"),
        ("Alcohol benefits the liver.", "Excessive alcohol damages the liver.", "hallucination"),
        ("Diabetes lowers blood sugar only.", "Diabetes involves blood sugar dysregulation.", "hallucination"),
        ("Blood pressure is irrelevant to health.", "Blood pressure indicates cardiovascular health.", "hallucination"),
        ("Cholesterol has no health effects.", "Cholesterol affects heart disease risk.", "hallucination"),
        ("Cancer is normal cell growth.", "Cancer involves abnormal uncontrolled growth.", "hallucination"),
        ("Inflammation is always harmful.", "Inflammation is a necessary immune response.", "hallucination"),
        ("Pain has no purpose.", "Pain signals tissue damage as warning.", "hallucination"),
        ("Fever is always dangerous.", "Fever is a natural infection-fighting response.", "hallucination"),
        ("Allergies are beneficial.", "Allergies are problematic immune overreactions.", "hallucination"),
        ("Asthma improves breathing.", "Asthma restricts breathing and airways.", "hallucination"),
        ("Osteoporosis strengthens bones.", "Osteoporosis weakens and thins bones.", "hallucination"),
        ("Arthritis has no symptoms.", "Arthritis causes significant joint pain.", "hallucination"),
        ("Alzheimer's improves memory.", "Alzheimer's severely impairs memory.", "hallucination"),
        ("Depression causes happiness.", "Depression causes persistent sadness.", "hallucination"),
        ("Anxiety eliminates worry.", "Anxiety causes excessive worry.", "hallucination"),
        ("Hygiene spreads disease.", "Hygiene prevents disease transmission.", "hallucination"),
        ("Quarantine spreads infection.", "Quarantine limits infection spread.", "hallucination"),
        ("Pandemics are local only.", "Pandemics are widespread global outbreaks.", "hallucination"),

        # WEATHER & CLIMATE (50 pairs)
        # Factual
        ("Rain forms from water vapor.", "Condensation creates precipitation.", "factual"),
        ("Snow is frozen precipitation.", "Ice crystals fall as snow.", "factual"),
        ("Humidity measures air moisture.", "Water vapor content defines humidity.", "factual"),
        ("Wind is moving air.", "Air pressure differences create wind.", "factual"),
        ("Clouds form from condensation.", "Water vapor becomes visible droplets.", "factual"),
        ("Thunder follows lightning.", "Electrical discharge creates sound.", "factual"),
        ("Hurricanes are tropical storms.", "Intense rotating storms form over oceans.", "factual"),
        ("Tornadoes are rotating columns.", "Violent vortexes extend from clouds.", "factual"),
        ("Droughts are prolonged dryness.", "Extended periods lack precipitation.", "factual"),
        ("Floods occur from excessive rain.", "Water overflow submerges land.", "factual"),
        ("Hail is frozen rain.", "Ice pellets form in thunderstorms.", "factual"),
        ("Fog is low-lying cloud.", "Ground-level condensation reduces visibility.", "factual"),
        ("Frost forms on cold surfaces.", "Ice crystals deposit on objects.", "factual"),
        ("Dew forms from condensation.", "Water droplets appear on surfaces.", "factual"),
        ("Climate differs from weather.", "Long-term patterns versus short-term conditions.", "factual"),
        ("Global warming raises temperatures.", "Earth's average temperature increases.", "factual"),
        ("Greenhouse gases trap heat.", "Atmospheric gases retain thermal energy.", "factual"),
        ("Carbon dioxide contributes to warming.", "CO2 emissions affect climate.", "factual"),
        ("Seasons result from Earth's tilt.", "Axial inclination causes seasonal change.", "factual"),
        ("The atmosphere protects Earth.", "Gas layers shield from harmful radiation.", "factual"),
        ("Barometric pressure indicates weather.", "Air pressure predicts conditions.", "factual"),
        ("Cold fronts bring temperature drops.", "Cool air masses lower temperatures.", "factual"),
        ("Warm fronts raise temperatures.", "Warm air masses increase heat.", "factual"),
        ("Precipitation includes all falling water.", "Rain, snow, sleet, and hail collectively.", "factual"),
        ("Evaporation converts liquid to vapor.", "Water transitions to gas phase.", "factual"),
        
        # Hallucination
        ("Rain falls upward.", "Rain falls downward due to gravity.", "hallucination"),
        ("Snow is hot.", "Snow is frozen and cold.", "hallucination"),
        ("Humidity measures temperature.", "Humidity measures air moisture content.", "hallucination"),
        ("Wind is stationary air.", "Wind is moving air by definition.", "hallucination"),
        ("Clouds are solid objects.", "Clouds are collections of water droplets.", "hallucination"),
        ("Thunder creates lightning.", "Lightning creates thunder, not vice versa.", "hallucination"),
        ("Hurricanes form in deserts.", "Hurricanes form over warm ocean waters.", "hallucination"),
        ("Tornadoes are gentle breezes.", "Tornadoes are violent rotating storms.", "hallucination"),
        ("Droughts have excessive water.", "Droughts involve lack of precipitation.", "hallucination"),
        ("Floods occur from no rain.", "Floods result from excessive water.", "hallucination"),
        ("Hail is liquid water.", "Hail consists of frozen ice pellets.", "hallucination"),
        ("Fog is clear visibility.", "Fog reduces visibility significantly.", "hallucination"),
        ("Frost forms in summer heat.", "Frost forms when surfaces freeze.", "hallucination"),
        ("Dew is frozen solid.", "Dew is liquid water droplets.", "hallucination"),
        ("Climate and weather are identical.", "Climate is long-term, weather is short-term.", "hallucination"),
        ("Global warming cools Earth.", "Global warming raises Earth's temperature.", "hallucination"),
        ("Greenhouse gases cool atmosphere.", "Greenhouse gases trap and retain heat.", "hallucination"),
        ("Carbon dioxide cools the planet.", "Carbon dioxide contributes to warming.", "hallucination"),
        ("Seasons result from distance from Sun.", "Seasons result from Earth's axial tilt.", "hallucination"),
        ("The atmosphere harms Earth.", "The atmosphere protects and supports life.", "hallucination"),
        ("Barometric pressure is irrelevant.", "Barometric pressure indicates weather patterns.", "hallucination"),
        ("Cold fronts raise temperatures.", "Cold fronts bring temperature drops.", "hallucination"),
        ("Warm fronts lower temperatures.", "Warm fronts raise temperatures.", "hallucination"),
        ("Precipitation is only rain.", "Precipitation includes rain, snow, sleet, hail.", "hallucination"),
        ("Evaporation freezes water.", "Evaporation converts liquid water to vapor.", "hallucination"),

        # LANGUAGE & LITERATURE (50 pairs)
        # Factual
        ("Nouns name people, places, or things.", "Substantives identify entities.", "factual"),
        ("Verbs express actions or states.", "Words describe what subjects do.", "factual"),
        ("Adjectives describe nouns.", "Modifiers provide noun characteristics.", "factual"),
        ("Adverbs modify verbs.", "Words describe how actions occur.", "factual"),
        ("Metaphors compare without 'like'.", "Direct comparisons create imagery.", "factual"),
        ("Similes use 'like' or 'as'.", "Explicit comparisons link concepts.", "factual"),
        ("Alliteration repeats initial sounds.", "Consonant repetition creates effect.", "factual"),
        ("Rhyme creates sound patterns.", "Similar ending sounds match.", "factual"),
        ("Poetry uses artistic language.", "Verse employs creative expression.", "factual"),
        ("Prose is ordinary writing.", "Non-poetic text forms prose.", "factual"),
        ("Novels are long fictional works.", "Extended narrative literature.", "factual"),
        ("Short stories are brief narratives.", "Concise fictional tales.", "factual"),
        ("Essays present arguments.", "Written compositions explore topics.", "factual"),
        ("Biographies recount lives.", "Life stories of real people.", "factual"),
        ("Autobiographies are self-written.", "Authors describe their own lives.", "factual"),
        ("Fiction is imagined narrative.", "Creative invented stories.", "factual"),
        ("Nonfiction presents factual information.", "True accounts and explanations.", "factual"),
        ("Drama involves staged performance.", "Theatrical works for acting.", "factual"),
        ("Comedy aims to amuse.", "Humorous works entertain.", "factual"),
        ("Tragedy depicts serious themes.", "Sorrowful dramatic works.", "factual"),
        ("Characters are story figures.", "People or beings in narratives.", "factual"),
        ("Plot is story sequence.", "Events form narrative structure.", "factual"),
        ("Setting establishes time and place.", "Context locates narrative action.", "factual"),
        ("Theme conveys central message.", "Underlying meaning or idea.", "factual"),
        ("Dialogue represents speech.", "Characters' spoken words.", "factual"),
        
        # Hallucination
        ("Nouns describe actions.", "Nouns name entities; verbs describe actions.", "hallucination"),
        ("Verbs name places.", "Verbs express actions, not name places.", "hallucination"),
        ("Adjectives show actions.", "Adjectives describe nouns, not show actions.", "hallucination"),
        ("Adverbs name things.", "Adverbs modify verbs, don't name things.", "hallucination"),
        ("Metaphors use 'like' always.", "Metaphors compare directly without 'like'.", "hallucination"),
        ("Similes never use comparisons.", "Similes explicitly compare using 'like' or 'as'.", "hallucination"),
        ("Alliteration repeats meanings.", "Alliteration repeats initial sounds, not meanings.", "hallucination"),
        ("Rhyme has no sound patterns.", "Rhyme creates matching sound patterns.", "hallucination"),
        ("Poetry uses only facts.", "Poetry uses artistic creative language.", "hallucination"),
        ("Prose is always poetic.", "Prose is ordinary non-poetic writing.", "hallucination"),
        ("Novels are always short.", "Novels are extended long fictional works.", "hallucination"),
        ("Short stories are very long.", "Short stories are brief narratives.", "hallucination"),
        ("Essays are fictional stories.", "Essays present arguments and ideas.", "hallucination"),
        ("Biographies are fictional.", "Biographies recount real lives.", "hallucination"),
        ("Autobiographies are written by others.", "Autobiographies are self-written accounts.", "hallucination"),
        ("Fiction presents only facts.", "Fiction is imagined creative narrative.", "hallucination"),
        ("Nonfiction is always imagined.", "Nonfiction presents factual information.", "hallucination"),
        ("Drama is silent reading.", "Drama involves staged performance.", "hallucination"),
        ("Comedy is always sad.", "Comedy aims to amuse and entertain.", "hallucination"),
        ("Tragedy is humorous.", "Tragedy depicts serious sorrowful themes.", "hallucination"),
        ("Characters are real people only.", "Characters can be any story figures.", "hallucination"),
        ("Plot is story setting.", "Plot is the sequence of events.", "hallucination"),
        ("Setting describes characters.", "Setting establishes time and place.", "hallucination"),
        ("Theme is the title.", "Theme conveys the central message.", "hallucination"),
        ("Dialogue is silent thought only.", "Dialogue represents spoken words.", "hallucination"),

        # PSYCHOLOGY & BEHAVIOR (40 pairs)
        # Factual
        ("Memory stores information.", "Brain retains learned experiences.", "factual"),
        ("Perception interprets sensory input.", "Mind processes environmental stimuli.", "factual"),
        ("Emotions influence behavior.", "Feelings affect decision-making.", "factual"),
        ("Learning changes behavior.", "Experience modifies responses.", "factual"),
        ("Motivation drives actions.", "Internal forces prompt behavior.", "factual"),
        ("Attention focuses awareness.", "Selective concentration on stimuli.", "factual"),
        ("Cognition involves thinking.", "Mental processes include reasoning.", "factual"),
        ("Personality reflects consistent traits.", "Individual characteristics persist.", "factual"),
        ("Consciousness is awareness.", "Being awake and perceiving.", "factual"),
        ("Habits are repeated behaviors.", "Regular patterns form routines.", "factual"),
        ("Social influence affects choices.", "Others impact our decisions.", "factual"),
        ("Development occurs throughout life.", "Growth continues from birth to death.", "factual"),
        ("Intelligence involves problem-solving.", "Cognitive ability to reason.", "factual"),
        ("Language enables communication.", "Symbolic systems convey meaning.", "factual"),
        ("Sleep consolidates memories.", "Rest strengthens learned information.", "factual"),
        ("Dreams occur during sleep.", "Mental imagery happens while resting.", "factual"),
        ("Stress triggers physical responses.", "Body reacts to perceived threats.", "factual"),
        ("Conditioning shapes behavior.", "Associations modify responses.", "factual"),
        ("Rewards reinforce actions.", "Positive outcomes increase behaviors.", "factual"),
        ("Punishment discourages behaviors.", "Negative consequences reduce actions.", "factual"),
        
        # Hallucination
        ("Memory deletes all information.", "Memory stores and retains information.", "hallucination"),
        ("Perception ignores all input.", "Perception interprets sensory information.", "hallucination"),
        ("Emotions have no effect.", "Emotions significantly influence behavior.", "hallucination"),
        ("Learning never changes behavior.", "Learning modifies behavior through experience.", "hallucination"),
        ("Motivation prevents action.", "Motivation drives and prompts actions.", "hallucination"),
        ("Attention scatters awareness.", "Attention focuses and concentrates awareness.", "hallucination"),
        ("Cognition avoids thinking.", "Cognition fundamentally involves thinking.", "hallucination"),
        ("Personality constantly changes.", "Personality traits are relatively consistent.", "hallucination"),
        ("Consciousness is unawareness.", "Consciousness is awareness of self and environment.", "hallucination"),
        ("Habits are random actions.", "Habits are consistent repeated behaviors.", "hallucination"),
        ("Social influence has no effect.", "Social factors significantly affect choices.", "hallucination"),
        ("Development stops at birth.", "Development continues throughout lifespan.", "hallucination"),
        ("Intelligence avoids problem-solving.", "Intelligence involves reasoning and problem-solving.", "hallucination"),
        ("Language prevents communication.", "Language enables effective communication.", "hallucination"),
        ("Sleep erases all memories.", "Sleep actually consolidates memories.", "hallucination"),
        ("Dreams never occur.", "Dreams regularly occur during sleep.", "hallucination"),
        ("Stress has no physical effects.", "Stress triggers significant physical responses.", "hallucination"),
        ("Conditioning is impossible.", "Conditioning effectively shapes behavior.", "hallucination"),
        ("Rewards discourage actions.", "Rewards reinforce and increase behaviors.", "hallucination"),
        ("Punishment encourages behaviors.", "Punishment discourages and reduces actions.", "hallucination"),

        # ECONOMICS & BUSINESS (40 pairs)
        # Factual
        ("Supply and demand determine prices.", "Market forces set economic values.", "factual"),
        ("Competition drives innovation.", "Rivalry encourages improvement.", "factual"),
        ("Inflation reduces purchasing power.", "Money buys less over time.", "factual"),
        ("Interest is cost of borrowing.", "Lenders charge for money use.", "factual"),
        ("Profit is revenue minus costs.", "Earnings after expenses.", "factual"),
        ("Markets facilitate exchange.", "Systems enable trade.", "factual"),
        ("Currency enables transactions.", "Money facilitates purchases.", "factual"),
        ("Banks store and lend money.", "Financial institutions manage funds.", "factual"),
        ("Investments seek returns.", "Capital deployment aims for gains.", "factual"),
        ("Taxes fund government services.", "Collected revenue supports programs.", "factual"),
        ("Budgets plan spending.", "Financial plans allocate resources.", "factual"),
        ("Debt is borrowed money.", "Obligations require repayment.", "factual"),
        ("Credit allows delayed payment.", "Purchasing power before payment.", "factual"),
        ("Savings accumulate wealth.", "Reserved funds build assets.", "factual"),
        ("Trade involves exchanging goods.", "Commerce requires mutual exchange.", "factual"),
        ("Exports leave the country.", "Goods sold internationally.", "factual"),
        ("Imports enter the country.", "Goods purchased from abroad.", "factual"),
        ("Entrepreneurs start businesses.", "Individuals create enterprises.", "factual"),
        ("Marketing promotes products.", "Strategies attract customers.", "factual"),
        ("Brands identify companies.", "Names distinguish businesses.", "factual"),
        
        # Hallucination
        ("Supply and demand are unrelated to prices.", "Supply and demand directly determine prices.", "hallucination"),
        ("Competition prevents innovation.", "Competition actually drives innovation.", "hallucination"),
        ("Inflation increases purchasing power.", "Inflation reduces what money can buy.", "hallucination"),
        ("Interest is free money.", "Interest is the cost of borrowing money.", "hallucination"),
        ("Profit is revenue only.", "Profit is revenue minus all costs.", "hallucination"),
        ("Markets prevent exchange.", "Markets facilitate trade and exchange.", "hallucination"),
        ("Currency prevents transactions.", "Currency enables and facilitates transactions.", "hallucination"),
        ("Banks destroy money.", "Banks store, lend, and manage money.", "hallucination"),
        ("Investments guarantee losses.", "Investments seek positive returns.", "hallucination"),
        ("Taxes are optional donations.", "Taxes are mandatory payments for services.", "hallucination"),
        ("Budgets ignore spending.", "Budgets specifically plan spending.", "hallucination"),
        ("Debt is free money.", "Debt must be repaid with interest.", "hallucination"),
        ("Credit requires immediate payment.", "Credit allows delayed payment.", "hallucination"),
        ("Savings lose wealth.", "Savings accumulate and build wealth.", "hallucination"),
        ("Trade avoids exchanging goods.", "Trade fundamentally involves exchange.", "hallucination"),
        ("Exports enter the country.", "Exports leave the country for sale abroad.", "hallucination"),
        ("Imports leave the country.", "Imports enter from foreign sources.", "hallucination"),
        ("Entrepreneurs close businesses.", "Entrepreneurs start and create businesses.", "hallucination"),
        ("Marketing hides products.", "Marketing promotes and showcases products.", "hallucination"),
        ("Brands confuse companies.", "Brands identify and distinguish companies.", "hallucination"),

        # SPORTS & RECREATION (40 pairs)
        # Factual
        ("Athletes compete in sports.", "Participants engage in physical contests.", "factual"),
        ("Training improves performance.", "Practice enhances athletic skills.", "factual"),
        ("Teams work collaboratively.", "Groups coordinate toward goals.", "factual"),
        ("Rules govern gameplay.", "Regulations structure competition.", "factual"),
        ("Referees enforce rules.", "Officials ensure fair play.", "factual"),
        ("Scoring determines winners.", "Points decide competition outcomes.", "factual"),
        ("Stamina enables endurance.", "Physical capacity sustains effort.", "factual"),
        ("Strategy guides tactics.", "Planning directs competitive approach.", "factual"),
        ("Equipment assists performance.", "Tools enhance athletic ability.", "factual"),
        ("Leagues organize competitions.", "Structured groups arrange matches.", "factual"),
        ("Championships award titles.", "Tournaments crown victors.", "factual"),
        ("Records track achievements.", "Statistics document performances.", "factual"),
        ("Warm-ups prepare athletes.", "Pre-activity routines ready body.", "factual"),
        ("Cool-downs aid recovery.", "Post-activity helps muscles recuperate.", "factual"),
        ("Injuries require rest.", "Damage needs healing time.", "factual"),
        ("Nutrition fuels athletes.", "Proper diet supports performance.", "factual"),
        ("Coaches provide guidance.", "Trainers instruct and mentor.", "factual"),
        ("Fans support teams.", "Spectators cheer for athletes.", "factual"),
        ("Olympic Games are international.", "Global athletes compete periodically.", "factual"),
        ("Sportsmanship promotes respect.", "Ethical conduct honors competition.", "factual"),
        
        # Hallucination
        ("Athletes avoid sports.", "Athletes actively compete in sports.", "hallucination"),
        ("Training worsens performance.", "Training improves athletic performance.", "hallucination"),
        ("Teams work independently only.", "Teams collaborate and work together.", "hallucination"),
        ("Rules are unnecessary.", "Rules are essential for fair gameplay.", "hallucination"),
        ("Referees ignore rules.", "Referees enforce rules during games.", "hallucination"),
        ("Scoring is irrelevant.", "Scoring determines competition winners.", "hallucination"),
        ("Stamina prevents endurance.", "Stamina enables sustained endurance.", "hallucination"),
        ("Strategy is random.", "Strategy involves purposeful planning.", "hallucination"),
        ("Equipment hinders performance.", "Equipment assists and enhances performance.", "hallucination"),
        ("Leagues prevent competitions.", "Leagues organize structured competitions.", "hallucination"),
        ("Championships have no winners.", "Championships award titles to winners.", "hallucination"),
        ("Records are ignored.", "Records track and document achievements.", "hallucination"),
        ("Warm-ups harm athletes.", "Warm-ups prepare and protect athletes.", "hallucination"),
        ("Cool-downs prevent recovery.", "Cool-downs facilitate muscle recovery.", "hallucination"),
        ("Injuries require more stress.", "Injuries require rest for healing.", "hallucination"),
        ("Nutrition harms athletes.", "Proper nutrition fuels athletic performance.", "hallucination"),
        ("Coaches confuse players.", "Coaches provide guidance and instruction.", "hallucination"),
        ("Fans oppose teams.", "Fans support and cheer for teams.", "hallucination"),
        ("Olympic Games are local only.", "Olympic Games are international events.", "hallucination"),
        ("Sportsmanship promotes cheating.", "Sportsmanship promotes respect and ethics.", "hallucination"),

        # ENVIRONMENT & ECOLOGY (50 pairs)
        # Factual
        ("Ecosystems balance naturally.", "Environmental systems maintain equilibrium.", "factual"),
        ("Pollution harms environments.", "Contamination damages ecosystems.", "factual"),
        ("Recycling reduces waste.", "Reprocessing conserves resources.", "factual"),
        ("Deforestation removes trees.", "Clearing destroys forest habitats.", "factual"),
        ("Endangered species face extinction.", "Threatened animals risk disappearing.", "factual"),
        ("Conservation protects nature.", "Preservation maintains biodiversity.", "factual"),
        ("Renewable energy is sustainable.", "Replenishable sources continue indefinitely.", "factual"),
        ("Fossil fuels are finite.", "Non-renewable resources will deplete.", "factual"),
        ("Solar power uses sunlight.", "Photovoltaic systems capture radiation.", "factual"),
        ("Wind turbines generate electricity.", "Air movement produces power.", "factual"),
        ("Hydropower uses water flow.", "Moving water creates energy.", "factual"),
        ("Carbon footprint measures emissions.", "Individual impact on atmosphere.", "factual"),
        ("Climate change affects weather patterns.", "Global shifts alter conditions.", "factual"),
        ("Melting ice raises sea levels.", "Glacial loss increases ocean height.", "factual"),
        ("Habitat loss threatens species.", "Environment destruction endangers wildlife.", "factual"),
        ("Overfishing depletes populations.", "Excessive harvesting reduces stocks.", "factual"),
        ("Pesticides harm insects.", "Chemicals kill targeted pests.", "factual"),
        ("Composting enriches soil.", "Organic breakdown adds nutrients.", "factual"),
        ("Water conservation preserves resources.", "Reducing use maintains supplies.", "factual"),
        ("Biodegradable materials decompose.", "Natural breakdown occurs over time.", "factual"),
        ("Plastic persists in environment.", "Synthetic materials don't decompose easily.", "factual"),
        ("Sustainable practices protect future.", "Responsible actions preserve resources.", "factual"),
        ("Invasive species disrupt ecosystems.", "Non-native organisms harm balance.", "factual"),
        ("Natural disasters impact environment.", "Catastrophic events alter landscapes.", "factual"),
        ("Urban sprawl reduces wildlife habitat.", "City expansion encroaches on nature.", "factual"),
        
        # Hallucination
        ("Ecosystems never balance.", "Ecosystems naturally seek equilibrium.", "hallucination"),
        ("Pollution improves environments.", "Pollution harms and damages ecosystems.", "hallucination"),
        ("Recycling increases waste.", "Recycling reduces and manages waste.", "hallucination"),
        ("Deforestation plants trees.", "Deforestation removes and destroys trees.", "hallucination"),
        ("Endangered species are thriving.", "Endangered species face extinction risk.", "hallucination"),
        ("Conservation destroys nature.", "Conservation protects and preserves nature.", "hallucination"),
        ("Renewable energy depletes quickly.", "Renewable energy is sustainable long-term.", "hallucination"),
        ("Fossil fuels are infinite.", "Fossil fuels are finite and will run out.", "hallucination"),
        ("Solar power uses no sunlight.", "Solar power specifically captures sunlight.", "hallucination"),
        ("Wind turbines consume electricity.", "Wind turbines generate electricity from wind.", "hallucination"),
        ("Hydropower uses no water.", "Hydropower uses water flow for energy.", "hallucination"),
        ("Carbon footprint measures health.", "Carbon footprint measures emissions impact.", "hallucination"),
        ("Climate change stabilizes weather.", "Climate change disrupts weather patterns.", "hallucination"),
        ("Melting ice lowers sea levels.", "Melting ice raises sea levels.", "hallucination"),
        ("Habitat loss helps species.", "Habitat loss threatens species survival.", "hallucination"),
        ("Overfishing increases populations.", "Overfishing depletes fish populations.", "hallucination"),
        ("Pesticides nurture insects.", "Pesticides are designed to kill insects.", "hallucination"),
        ("Composting depletes soil.", "Composting enriches and improves soil.", "hallucination"),
        ("Water conservation wastes resources.", "Water conservation preserves resources.", "hallucination"),
        ("Biodegradable materials persist forever.", "Biodegradable materials decompose naturally.", "hallucination"),
        ("Plastic decomposes immediately.", "Plastic persists without easy decomposition.", "hallucination"),
        ("Sustainable practices deplete future.", "Sustainable practices protect the future.", "hallucination"),
        ("Invasive species help ecosystems.", "Invasive species disrupt ecosystem balance.", "hallucination"),
        ("Natural disasters improve environment.", "Natural disasters can severely impact environment.", "hallucination"),
        ("Urban sprawl increases wildlife habitat.", "Urban sprawl reduces natural habitat.", "hallucination"),
    ]
    
    pairs.extend(synthetic_pairs)
    dataset_stats["synthetic"] = len(synthetic_pairs)
    
    # Load TruthfulQA dataset
    if "truthfulqa" in config.datasets:
        try:
            logger.log("Loading TruthfulQA dataset...")
            truthfulqa = datasets.load_dataset("domenicrosati/TruthfulQA", split="train")
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
    
    # Try to load FEVER dataset if requested
    if "fever" in config.datasets:
        try:
            logger.log("Attempting to load FEVER dataset...")
            fever = datasets.load_dataset("copenlu/fever_gold_evidence", split="train[:500]")
            fever_pairs = []
            for ex in fever:
                lab = (ex.get("label") or "").upper()
                if lab not in ("SUPPORTS","REFUTES"): 
                    continue
                ev = ex.get("evidence_text") or " ".join(
                    [e["text"] for e in ex.get("evidence",[]) if isinstance(e,dict) and "text" in e]
                )
                if not ev or len(ev.split()) < 3: 
                    continue
                fever_pairs.append((ex["claim"], ev, "factual" if lab=="SUPPORTS" else "hallucination"))
            
            if fever_pairs:
                pairs.extend(fever_pairs)
                dataset_stats["fever"] = len(fever_pairs)
                logger.log(f"Loaded FEVER: {len(fever_pairs)} pairs")
            else:
                logger.log("FEVER dataset loaded but no valid pairs found")
                
        except Exception as e:
            logger.log(f"FEVER load failed: {e}")

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
# Fixed Visualization Functions
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

def plot_alignment_heatmaps(layerwise_data):
    """Plot heatmaps of alignment trajectories across layers"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    if "factual" in layerwise_data and layerwise_data["factual"]:
        factual_matrix = np.array(layerwise_data["factual"])
        im1 = ax1.imshow(factual_matrix, aspect='auto', cmap='RdYlGn', 
                        extent=[0, factual_matrix.shape[1], 0, factual_matrix.shape[0]])
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Sample Index')
        ax1.set_title('Factual Alignment Trajectories')
        plt.colorbar(im1, ax=ax1, label='Alignment Score')
    
    if "hallucination" in layerwise_data and layerwise_data["hallucination"]:
        hallucination_matrix = np.array(layerwise_data["hallucination"])
        im2 = ax2.imshow(hallucination_matrix, aspect='auto', cmap='RdYlGn',
                         extent=[0, hallucination_matrix.shape[1], 0, hallucination_matrix.shape[0]])
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Sample Index')
        ax2.set_title('Hallucination Alignment Trajectories')
        plt.colorbar(im2, ax=ax2, label='Alignment Score')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "alignment_heatmaps.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_trajectory_clusters(layerwise_data):
    """Cluster and visualize different trajectory patterns"""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    
    # Combine all trajectories
    all_trajectories = []
    labels = []
    
    if "factual" in layerwise_data:
        all_trajectories.extend(layerwise_data["factual"])
        labels.extend(['factual'] * len(layerwise_data["factual"]))
    
    if "hallucination" in layerwise_data:
        all_trajectories.extend(layerwise_data["hallucination"])
        labels.extend(['hallucination'] * len(layerwise_data["hallucination"]))
    
    if not all_trajectories:
        return
        
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    trajectories_2d = pca.fit_transform(all_trajectories)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot by ground truth
    colors = {'factual': 'green', 'hallucination': 'red'}
    for i, label in enumerate(labels):
        ax1.scatter(trajectories_2d[i, 0], trajectories_2d[i, 1], 
                   c=colors[label], alpha=0.6, label=label if i == 0 else "")
    
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('Trajectory Patterns by Ground Truth')
    ax1.legend()
    
    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(all_trajectories)
    
    scatter = ax2.scatter(trajectories_2d[:, 0], trajectories_2d[:, 1], 
                         c=cluster_labels, cmap='viridis', alpha=0.6)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('Trajectory Patterns by K-means Clustering')
    plt.colorbar(scatter, ax=ax2, label='Cluster')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "trajectory_clusters.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_early_late_dynamics(df, layerwise_data):
    """Compare early vs late layer behaviors"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Early vs late alignment ratios
    if "factual" in layerwise_data and "hallucination" in layerwise_data:
        factual_trajs = np.array(layerwise_data["factual"])
        hallucination_trajs = np.array(layerwise_data["hallucination"])
        
        # Calculate early (first 25%) vs late (last 25%) alignment
        early_cutoff = max(1, factual_trajs.shape[1] // 4)
        late_start = factual_trajs.shape[1] - early_cutoff
        
        factual_early = factual_trajs[:, :early_cutoff].mean(axis=1)
        factual_late = factual_trajs[:, late_start:].mean(axis=1)
        hallucination_early = hallucination_trajs[:, :early_cutoff].mean(axis=1)
        hallucination_late = hallucination_trajs[:, late_start:].mean(axis=1)
        
        # Early layer comparison
        axes[0,0].boxplot([factual_early, hallucination_early], 
                         labels=['Factual', 'Hallucination'])
        axes[0,0].set_title('Early Layer Alignment (First 25%)')
        axes[0,0].set_ylabel('Alignment Score')
        
        # Late layer comparison
        axes[0,1].boxplot([factual_late, hallucination_late], 
                         labels=['Factual', 'Hallucination'])
        axes[0,1].set_title('Late Layer Alignment (Last 25%)')
        axes[0,1].set_ylabel('Alignment Score')
        
        # Improvement ratio
        factual_improvement = factual_late / factual_early
        hallucination_improvement = hallucination_late / hallucination_early
        
        axes[1,0].boxplot([factual_improvement, hallucination_improvement],
                         labels=['Factual', 'Hallucination'])
        axes[1,0].set_title('Alignment Improvement Ratio (Late/Early)')
        axes[1,0].set_ylabel('Improvement Ratio')
        axes[1,0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
    
    # Convergence speed
    if "convergence_layer" in df.columns and "num_layers" in df.columns:
        convergence_speed = df["convergence_layer"] / df["num_layers"]
        
        sns.violinplot(data=df, x="label", y=convergence_speed, hue="label",
                      palette={"factual": "green", "hallucination": "red"}, 
                      ax=axes[1,1], legend=False)
        axes[1,1].set_title('Normalized Convergence Layer\n(Lower = Faster Convergence)')
        axes[1,1].set_ylabel('Convergence Layer / Total Layers')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "early_late_dynamics.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_analysis(df):
    """Plot correlation matrices between different metrics"""
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_data = df[numeric_cols]
    
    # Separate by label
    factual_corr = df[df['label']=='factual'][numeric_cols].corr()
    hallucination_corr = df[df['label']=='hallucination'][numeric_cols].corr()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Overall correlation
    mask = np.triu(np.ones_like(correlation_data.corr(), dtype=bool))
    sns.heatmap(correlation_data.corr(), mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, ax=ax1, cbar_kws={'shrink': 0.8})
    ax1.set_title('Overall Correlation Matrix')
    
    # Factual correlation
    sns.heatmap(factual_corr, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, ax=ax2, cbar_kws={'shrink': 0.8})
    ax2.set_title('Factual Samples Correlation')
    
    # Hallucination correlation
    sns.heatmap(hallucination_corr, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, ax=ax3, cbar_kws={'shrink': 0.8})
    ax3.set_title('Hallucination Samples Correlation')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "correlation_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_smoothness_analysis(layerwise_data):
    """Analyze smoothness and stability of trajectories"""
    def calculate_smoothness(trajectory):
        """Calculate smoothness as inverse of mean squared second derivative"""
        if len(trajectory) < 3:
            return 0
        second_deriv = np.diff(trajectory, 2)
        return 1 / (1 + np.mean(second_deriv**2))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Smoothness distribution
    smoothness_data = {'factual': [], 'hallucination': []}
    
    for label in ['factual', 'hallucination']:
        if label in layerwise_data:
            smoothness_data[label] = [calculate_smoothness(traj) for traj in layerwise_data[label]]
    
    axes[0].boxplot([smoothness_data['factual'], smoothness_data['hallucination']],
                   labels=['Factual', 'Hallucination'])
    axes[0].set_title('Trajectory Smoothness\n(Higher = Smother)')
    axes[0].set_ylabel('Smoothness Score')
    
    # Example trajectories
    if "factual" in layerwise_data and "hallucination" in layerwise_data:
        # Plot smoothest and roughest examples
        factual_smoothness = smoothness_data['factual']
        hallucination_smoothness = smoothness_data['hallucination']
        
        factual_smoothest_idx = np.argmax(factual_smoothness)
        factual_roughest_idx = np.argmin(factual_smoothness)
        hallucination_smoothest_idx = np.argmax(hallucination_smoothness)
        hallucination_roughest_idx = np.argmin(hallucination_smoothness)
        
        axes[1].plot(layerwise_data['factual'][factual_smoothest_idx], 'g-', 
                    label='Factual Smoothest', linewidth=2)
        axes[1].plot(layerwise_data['factual'][factual_roughest_idx], 'g--', 
                    label='Factual Roughest', alpha=0.7)
        axes[1].plot(layerwise_data['hallucination'][hallucination_smoothest_idx], 'r-', 
                    label='Hallucination Smoothest', linewidth=2)
        axes[1].plot(layerwise_data['hallucination'][hallucination_roughest_idx], 'r--', 
                    label='Hallucination Roughest', alpha=0.7)
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('Alignment Score')
        axes[1].set_title('Extreme Smoothness Examples')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Smoothness vs final alignment scatter
    if "factual" in layerwise_data:
        factual_final_align = [traj[-1] for traj in layerwise_data['factual']]
        axes[2].scatter(smoothness_data['factual'], factual_final_align, 
                       alpha=0.6, color='green', label='Factual')
    
    if "hallucination" in layerwise_data:
        hallucination_final_align = [traj[-1] for traj in layerwise_data['hallucination']]
        axes[2].scatter(smoothness_data['hallucination'], hallucination_final_align, 
                       alpha=0.6, color='red', label='Hallucination')
    
    axes[2].set_xlabel('Smoothness Score')
    axes[2].set_ylabel('Final Alignment')
    axes[2].set_title('Smoothness vs Final Alignment')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "smoothness_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()



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
    logger.log("Starting Enhanced Layer-wise Semantic Dynamics Analysis")
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
        
        # Step 5: Comprehensive Evaluation (NEW)
        logger.log("Step 5: Comprehensive evaluation with metrics...")
        results, X, y = comprehensive_evaluation(df, layerwise_data, ex, te, h_proj, t_proj)
        
        # Step 6: Visualization (NEW)
        logger.log("Step 6: Generating comprehensive visualizations...")
        plot_comprehensive_metrics(results)
        
        # Step 7: Generate report (NEW)
        logger.log("Step 7: Generating evaluation report...")
        report = generate_evaluation_report(results, df)
        
        # Step 8: Original visualizations
        logger.log("Step 8: Generating original LSD visualizations...")
        plot_convergence(all_traj, stats_summary)
        plot_velocity_acceleration(df)
        plot_alignment_heatmaps(layerwise_data)
        plot_trajectory_clusters(layerwise_data)
        plot_early_late_dynamics(df, layerwise_data)
        plot_correlation_analysis(df)
        plot_smoothness_analysis(layerwise_data)
        
        # Final summary
        logger.log("ANALYSIS COMPLETED SUCCESSFULLY")
        
        # Print comprehensive results
        print("\n" + "="*70)
        print("LAYER-WISE SEMANTIC DYNAMICS ANALYSIS COMPLETE")
        print("="*70)
        
        print(f"\nCOMPREHENSIVE EVALUATION RESULTS:")
        print("-" * 50)
        for method_name, result in results.items():
            print(f"{method_name}:")
            print(f"  F1: {result['f1']:.4f} | AUC-ROC: {result['auroc']:.4f} | Specificity: {result['specificity']:.4f}")
        
        return results, report, df
        
    except Exception as e:
        logger.log(f"ERROR in main execution: {e}")
        import traceback
        logger.log(f"Traceback: {traceback.format_exc()}")
        print(f"Error occurred: {e}")
        print("Check the log file for details.")
        return None, None, None

if __name__ == "__main__":
    results, report, df = main()