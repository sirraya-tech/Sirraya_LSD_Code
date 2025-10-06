import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from typing import Dict, Any
from ..utils.helpers import logger

class StatisticalAnalyzer:
    """Enhanced statistical analysis for layer-wise semantic dynamics."""
    
    def __init__(self):
        self.alpha = 0.05
    
    def comprehensive_analysis(self, df: pd.DataFrame, layerwise_data: Dict) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
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
        """Compare factual vs hallucination groups using t-tests."""
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
        """Analyze statistical significance across layers."""
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
        """Generate statistical summary."""
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