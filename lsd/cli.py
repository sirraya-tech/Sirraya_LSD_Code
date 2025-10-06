#!/usr/bin/env python3
"""
Command Line Interface for Layer-wise Semantic Dynamics Analysis
"""

import argparse
import sys
from lsd.core import AnalysisOrchestrator, LayerwiseSemanticDynamicsConfig, OperationMode

def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Layer-wise Semantic Dynamics Analysis for Hallucination Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        "--num-pairs", type=int, default=1000,
        help="Number of text-truth pairs to generate"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["synthetic", "truthfulqa"],
        choices=["synthetic", "truthfulqa", "fever", "custom"],
        help="Datasets to use for training and evaluation"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-name", type=str, default="gpt2",
        help="Base language model to analyze"
    )
    parser.add_argument(
        "--truth-encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer for truth encoding"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-5,
        help="Learning rate for training"
    )
    
    # Mode arguments
    parser.add_argument(
        "--mode", type=str, default="hybrid",
        choices=["supervised", "unsupervised", "hybrid"],
        help="Analysis mode"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir", type=str, default="lsd_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--save-models", action="store_true",
        help="Save trained models"
    )
    
    # Utility arguments
    parser.add_argument(
        "--use-pretrained", action="store_true",
        help="Use pretrained models if available"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    
    return parser

def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = LayerwiseSemanticDynamicsConfig(
            model_name=args.model_name,
            truth_encoder_name=args.truth_encoder,
            num_pairs=args.num_pairs,
            datasets=args.datasets,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            mode=OperationMode(args.mode),
            use_pretrained=args.use_pretrained,
        )
        
        # Create orchestrator and run analysis
        orchestrator = AnalysisOrchestrator(config)
        report = orchestrator.run_comprehensive_analysis()
        
        # Print summary
        if report and report.get('status') != 'error':
            print("\n" + "="*60)
            print("ANALYSIS COMPLETED SUCCESSFULLY")
            print("="*60)
            
            if 'key_findings' in report:
                kf = report['key_findings']
                print(f"Best Method: {kf.get('best_method', 'N/A')}")
                print(f"Composite Score: {kf.get('best_composite_score', 0):.4f}")
                print(f"Detection Quality: {kf.get('detection_quality', 'N/A')}")
            
            print(f"\nResults saved to: {config.output_dir}")
            sys.exit(0)
        else:
            print(f"Analysis failed: {report.get('message', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()