import os, math, random
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import LayerwiseSemanticDynamicsConfig, OperationMode
from ..models.manager import ModelManager, train_projection_heads
from ..data.manager import DataManager
from ..evaluation.evaluator import ComprehensiveEvaluator
from ..evaluation.statistics import StatisticalAnalyzer
from ..visualization.engine import VisualizationEngine
from ..utils.helpers import convert_to_json_serializable, safe_json_dump, DirectoryManager, EnhancedLogger

# Initialize global directory manager and logger
dir_manager = DirectoryManager()
logger = EnhancedLogger()

class AnalysisOrchestrator:
    """
    Enhanced orchestrator with better error handling and diagnostics.
    Manages the complete analysis pipeline from data loading to final reporting.
    """
    
    def __init__(self, config: LayerwiseSemanticDynamicsConfig):
        self.config = config
        self.data_manager = DataManager(config)
        self.evaluator = ComprehensiveEvaluator(config)
        self.visualizer = VisualizationEngine()
        self.stat_analyzer = StatisticalAnalyzer()
        
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive layer-wise semantic dynamics analysis.
        
        Complete pipeline:
        1. Build dataset from multiple sources
        2. Train projection heads with contrastive learning
        3. Analyze layer-wise dynamics and extract features
        4. Perform statistical analysis
        5. Evaluate using supervised/unsupervised methods
        6. Generate visualizations
        7. Create comprehensive report
        
        Returns:
            Dictionary containing final analysis report
        """
        logger.log("=" * 80)
        logger.log("STARTING LAYER-WISE SEMANTIC DYNAMICS ANALYSIS")
        logger.log("=" * 80)
        
        try:
            # Step 1: Build dataset
            logger.log("\n[STEP 1/7] Building dataset...")
            pairs = self.data_manager.build_dataset()
            
            if not pairs:
                logger.log("ERROR: No pairs generated. Exiting.", "ERROR")
                return {"status": "error", "message": "No data pairs generated"}
            
            # Step 2: Enhanced training
            logger.log("\n[STEP 2/7] Running enhanced training...")
            model_manager = train_projection_heads(self.config, pairs)
            
            # Step 3: Analyze dynamics
            logger.log("\n[STEP 3/7] Analyzing layer-wise dynamics...")
            from ..models.feature_extractor import FeatureExtractor, analyze_layerwise_dynamics
            df, all_trajectories, layerwise_data = analyze_layerwise_dynamics(pairs, model_manager)
            
            if df.empty:
                logger.log("ERROR: No analysis results generated. Exiting.", "ERROR")
                return {"status": "error", "message": "Analysis produced no results"}
            
            # Step 4: Statistical analysis
            logger.log("\n[STEP 4/7] Performing statistical analysis...")
            stats_summary = self.stat_analyzer.comprehensive_analysis(df, layerwise_data)
            
            # Step 5: Evaluation based on mode
            logger.log(f"\n[STEP 5/7] Running {self.config.mode.value} evaluation...")
            
            if self.config.mode == OperationMode.SUPERVISED:
                evaluation_results = self.evaluator.evaluate_supervised(df)
            elif self.config.mode == OperationMode.UNSUPERVISED:
                evaluation_results = self.evaluator.evaluate_unsupervised(df)
            else:  # HYBRID
                evaluation_results = self.evaluator.evaluate_hybrid(df)
            
            # Step 6: Visualization
            logger.log("\n[STEP 6/7] Generating visualizations...")
            self.visualizer.plot_comprehensive_metrics(evaluation_results)
            
            # Step 7: Generate comprehensive report
            logger.log("\n[STEP 7/7] Generating final report...")
            final_report = self._generate_final_report(
                df, evaluation_results, stats_summary
            )
            
            # Save all results
            self._save_results(df, evaluation_results, stats_summary, final_report)
            
            logger.log("=" * 80)
            logger.log("COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY")
            logger.log("=" * 80)
            
            return final_report
            
        except Exception as e:
            logger.log(f"ERROR in comprehensive analysis: {e}", "ERROR")
            import traceback
            logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")
            return {"status": "error", "message": str(e)}
    
    def _generate_final_report(self, df: pd.DataFrame, evaluation_results: Dict,
                             stats_summary: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive final report.
        """
        report = {
            'execution_summary': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'config': {
                    'model_name': self.config.model_name,
                    'mode': self.config.mode.value,
                    'num_pairs': self.config.num_pairs
                },
                'device': str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                'total_samples': int(len(df))
            },
            'dataset_statistics': {
                'factual_samples': int(len(df[df['label'] == 'factual'])),
                'hallucination_samples': int(len(df[df['label'] == 'hallucination'])),
                'class_balance': float(len(df[df['label'] == 'factual']) / len(df)) if len(df) > 0 else 0.0
            },
            'evaluation_results': evaluation_results,
            'statistical_summary': stats_summary['summary'],
            'key_findings': {},
            'recommendations': []
        }
        
        # Extract key findings
        if 'supervised' in evaluation_results:
            supervised_results = evaluation_results['supervised']
            if supervised_results:
                best_method = max(supervised_results.items(), 
                                key=lambda x: x[1]['composite_score'])
                
                report['key_findings']['best_method'] = best_method[0]
                report['key_findings']['best_composite_score'] = float(best_method[1]['composite_score'])
                report['key_findings']['best_f1_score'] = float(best_method[1]['f1'])
                
                # Detection quality assessment
                composite_score = best_method[1]['composite_score']
                if composite_score >= 0.9:
                    detection_class = "EXCELLENT"
                elif composite_score >= 0.8:
                    detection_class = "VERY_GOOD" 
                elif composite_score >= 0.7:
                    detection_class = "GOOD"
                else:
                    detection_class = "MODERATE"
                    
                report['key_findings']['detection_quality'] = detection_class
        
        # Statistical findings
        stats_summary_data = stats_summary['summary']
        report['key_findings']['significant_metrics'] = int(stats_summary_data['significant_metrics_count'])
        
        # Generate recommendations
        self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: Dict[str, Any]):
        """Generate actionable recommendations based on analysis results."""
        recommendations = []
        
        # Based on detection quality
        detection_quality = report['key_findings'].get('detection_quality', 'MODERATE')
        if detection_quality == "EXCELLENT":
            recommendations.append("✓ Ready for production deployment in critical applications")
        elif detection_quality == "VERY_GOOD":
            recommendations.append("✓ Suitable for production deployment in most applications")
        elif detection_quality == "GOOD":
            recommendations.append("✓ Suitable for deployment with monitoring and validation")
        else:
            recommendations.append("⚠ Consider further optimization before production deployment")
        
        # Based on statistical findings
        sig_metrics = report['key_findings'].get('significant_metrics', 0)
        if sig_metrics >= 5:
            recommendations.append("✓ Strong statistical foundation with multiple significant metrics")
        elif sig_metrics >= 3:
            recommendations.append("✓ Good statistical foundation")
        else:
            recommendations.append("⚠ Limited statistical significance - consider more data")
        
        # Based on sample size
        total_samples = report['execution_summary']['total_samples']
        if total_samples < 100:
            recommendations.append("⚠ Small sample size - collect more data for robust results")
        elif total_samples < 500:
            recommendations.append("✓ Adequate sample size for initial validation")
        else:
            recommendations.append("✓ Good sample size for reliable analysis")
        
        report['recommendations'] = recommendations
    
    def _save_results(self, df: pd.DataFrame, evaluation_results: Dict,
                     stats_summary: Dict, final_report: Dict):
        """Save all results to files with proper JSON serialization."""
        try:
            # Save dataframe
            df.to_csv(dir_manager.results_dir / "final_analysis_results.csv", index=False)
            logger.log("Saved analysis results CSV")
            
            # Save evaluation results with type conversion
            safe_json_dump(
                evaluation_results,
                dir_manager.get_result_path("evaluation_results")
            )
            logger.log("Saved evaluation results JSON")
            
            # Save statistical summary with type conversion
            safe_json_dump(
                stats_summary,
                dir_manager.get_result_path("statistical_summary")
            )
            logger.log("Saved statistical summary JSON")
            
            # Save final report with type conversion
            safe_json_dump(
                final_report,
                dir_manager.get_result_path("final_report")
            )
            logger.log("Saved final report JSON")
            
            logger.log(f"All results saved to: {dir_manager.results_dir}")
            
        except Exception as e:
            logger.log(f"Error saving results: {e}", "ERROR")
            import traceback
            logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")