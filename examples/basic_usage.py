#!/usr/bin/env python3
"""
Basic usage example for Layer-wise Semantic Dynamics
"""

from lsd.core import AnalysisOrchestrator, LayerwiseSemanticDynamicsConfig, OperationMode

def main():
    """Basic example of using the LSD framework"""
    
    # Create configuration
    config = LayerwiseSemanticDynamicsConfig(
        model_name="gpt2",
        truth_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        num_pairs=200,  # Small dataset for quick testing
        epochs=5,
        batch_size=4,
        mode=OperationMode.HYBRID,
        use_pretrained=False
    )
    
    print("Starting Layer-wise Semantic Dynamics Analysis...")
    print(f"Model: {config.model_name}")
    print(f"Mode: {config.mode.value}")
    print(f"Samples: {config.num_pairs}")
    
    # Run analysis
    orchestrator = AnalysisOrchestrator(config)
    report = orchestrator.run_comprehensive_analysis()
    
    # Print results
    if report and report.get('status') != 'error':
        print("\n=== ANALYSIS RESULTS ===")
        print(f"Best Method: {report['key_findings']['best_method']}")
        print(f"Composite Score: {report['key_findings']['best_composite_score']:.4f}")
        print(f"Detection Quality: {report['key_findings']['detection_quality']}")
        print(f"Significant Metrics: {report['key_findings']['significant_metrics']}")
    else:
        print(f"Analysis failed: {report.get('message', 'Unknown error')}")

if __name__ == "__main__":
    main()