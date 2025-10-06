#!/usr/bin/env python3
"""
Advanced analysis example for Layer-wise Semantic Dynamics
"""

from lsd.core import AnalysisOrchestrator, LayerwiseSemanticDynamicsConfig, OperationMode

def main():
    """Advanced example with custom configuration"""
    
    # Custom configuration for advanced analysis
    config = LayerwiseSemanticDynamicsConfig(
        model_name="gpt2-medium",  # Larger model
        truth_encoder_name="sentence-transformers/all-mpnet-base-v2",  # Better encoder
        num_pairs=2000,
        epochs=20,
        batch_size=16,
        learning_rate=1e-5,
        mode=OperationMode.HYBRID,
        use_pretrained=False,
        enable_ensemble=True,
        composite_score_weights={
            'f1': 0.30,
            'auroc': 0.25,
            'precision': 0.15,
            'recall': 0.15,
            'specificity': 0.10,
            'mcc': 0.05
        }
    )
    
    print("Starting Advanced Layer-wise Semantic Dynamics Analysis...")
    print(f"Model: {config.model_name}")
    print(f"Truth Encoder: {config.truth_encoder_name}")
    print(f"Mode: {config.mode.value}")
    print(f"Samples: {config.num_pairs}")
    print(f"Epochs: {config.epochs}")
    
    # Run analysis
    orchestrator = AnalysisOrchestrator(config)
    report = orchestrator.run_comprehensive_analysis()
    
    # Detailed results
    if report and report.get('status') != 'error':
        print("\n=== ADVANCED ANALYSIS RESULTS ===")
        kf = report['key_findings']
        print(f"Best Method: {kf.get('best_method', 'N/A')}")
        print(f"Composite Score: {kf.get('best_composite_score', 0):.4f}")
        print(f"F1 Score: {kf.get('best_f1_score', 0):.4f}")
        print(f"AUC-ROC: {report['evaluation_results']['supervised'][kf['best_method']]['auroc']:.4f}")
        print(f"Detection Quality: {kf.get('detection_quality', 'N/A')}")
        print(f"Significant Metrics: {kf.get('significant_metrics', 0)}")
        
        # Show all classifier performances
        print(f"\nClassifier Performances:")
        for method, metrics in report['evaluation_results']['supervised'].items():
            print(f"  {method}: F1={metrics['f1']:.4f}, AUC={metrics['auroc']:.4f}")
            
    else:
        print(f"Analysis failed: {report.get('message', 'Unknown error')}")

if __name__ == "__main__":
    main()