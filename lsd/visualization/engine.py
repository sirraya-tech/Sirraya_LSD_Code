import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from typing import Dict, Any
from ..utils.helpers import logger, dir_manager

class VisualizationEngine:
    """Comprehensive visualization system for layer-wise semantic dynamics analysis."""
    
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
        """Plot comprehensive evaluation metrics across all classifiers."""
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