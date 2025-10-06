#!/usr/bin/env python3
"""
Main execution function for Layer-wise Semantic Dynamics Analysis
"""

from lsd.core import AnalysisOrchestrator, LayerwiseSemanticDynamicsConfig, OperationMode

def create_enhanced_config() -> LayerwiseSemanticDynamicsConfig:
    """
    Create enhanced detection configuration with optimized parameters.
    
    Returns:
        Configured LayerwiseSemanticDynamicsConfig object
    """
    return LayerwiseSemanticDynamicsConfig(
        model_name="gpt2",
        truth_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        num_pairs=1000,
        epochs=10,
        batch_size=8,
        learning_rate=5e-5,
        margin=0.5,
        mode=OperationMode.HYBRID,
        use_pretrained=False,
        enable_ensemble=True,
        composite_score_weights={
            'f1': 0.25,
            'auroc': 0.20,
            'precision': 0.15,
            'recall': 0.15,
            'specificity': 0.10,
            'mcc': 0.15
        }
    )

def main():
    """
    Main execution function.
    
    Orchestrates the complete layer-wise semantic dynamics analysis pipeline.
    """
    
    print("\n" + "="*80)
    print("LAYER-WISE SEMANTIC DYNAMICS ANALYSIS SYSTEM")
    print("Hallucination Detection via Semantic Trajectory Analysis")
    print("="*80 + "\n")
    
    # Use enhanced configuration
    config = create_enhanced_config()
    
    print(f"Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Mode: {config.mode.value}")
    print(f"  Samples: {config.num_pairs}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Device: CPU/GPU\n")
    
    # Create and run orchestrator
    orchestrator = AnalysisOrchestrator(config)
    final_report = orchestrator.run_comprehensive_analysis()
    
    # Print summary
    if final_report and 'status' not in final_report or final_report.get('status') != 'error':
        print("\n" + "="*80)
        print("LAYER-WISE SEMANTIC DYNAMICS - ANALYSIS COMPLETE")
        print("="*80)
        
        print(f"\nEXECUTION SUMMARY:")
        print(f"  Total samples analyzed: {final_report['execution_summary']['total_samples']}")
        print(f"  Operation mode: {config.mode.value}")
        print(f"  Device: {final_report['execution_summary']['device']}")
        print(f"  Timestamp: {final_report['execution_summary']['timestamp']}")
        
        if 'dataset_statistics' in final_report:
            ds = final_report['dataset_statistics']
            print(f"\nDATASET STATISTICS:")
            print(f"  Factual samples: {ds['factual_samples']}")
            print(f"  Hallucination samples: {ds['hallucination_samples']}")
            print(f"  Class balance: {ds['class_balance']:.2%}")
        
        if 'key_findings' in final_report:
            kf = final_report['key_findings']
            print(f"\nKEY FINDINGS:")
            print(f"  Best method: {kf.get('best_method', 'N/A')}")
            print(f"  Composite score: {kf.get('best_composite_score', 0):.4f}")
            print(f"  F1 score: {kf.get('best_f1_score', 0):.4f}")
            print(f"  Detection quality: {kf.get('detection_quality', 'N/A')}")
            print(f"  Significant metrics: {kf.get('significant_metrics', 0)}")
        
        print(f"\nRECOMMENDATIONS:")
        for rec in final_report.get('recommendations', []):
            print(f"  {rec}")
        
        print(f"\nRESULTS SAVED TO: layerwise_semantic_dynamics_system/")
        print("  - Models: models/")
        print("  - Plots: plots/")
        print("  - Results: results/")
        print("  - Logs: execution.log")
        print("="*80 + "\n")
        
        return final_report
    else:
        print("\n" + "="*80)
        print("ANALYSIS FAILED")
        print("="*80)
        print(f"Error: {final_report.get('message', 'Unknown error')}")
        print("Check the logs for details.")
        print("="*80 + "\n")
        return None

if __name__ == "__main__":
    main()