```markdown
# Layer-wise Semantic Dynamics (LSD) for Hallucination Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/arXiv-Paper-orange" alt="arXiv">
</p>

## 🎯 What is LSD?

**Layer-wise Semantic Dynamics (LSD)** is a geometric framework that detects hallucinations in Large Language Models by analyzing how semantic representations evolve across transformer layers. Unlike traditional methods that require multiple sampling passes or external knowledge bases, LSD operates intrinsically with **single-forward-pass efficiency**.

> **Key Insight**: Factual content follows smooth, convergent trajectories toward truth embeddings, while hallucinations exhibit oscillatory, divergent patterns.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sirraya-tech/Sirraya_LSD_Code.git
cd Sirraya_LSD_Code

# Install dependencies
pip install torch transformers sentence-transformers datasets scikit-learn matplotlib seaborn tqdm pandas numpy
```

### Basic Usage

```python
from lsd_framework import create_enhanced_config, main

# Run complete analysis pipeline
config = create_enhanced_config()
results = main()

# Or use specific components
from lsd_framework import AnalysisOrchestrator, LayerwiseSemanticDynamicsConfig

config = LayerwiseSemanticDynamicsConfig(
    model_name="gpt2",
    num_pairs=500,
    mode="hybrid"  # supervised, unsupervised, or hybrid
)

orchestrator = AnalysisOrchestrator(config)
report = orchestrator.run_comprehensive_analysis()
```

### One-Line Demo

```python
# Complete analysis with default settings
python -c "from lsd_framework import main; main()"
```

## 📊 Performance Highlights

*   **F1-Score**: 0.92
*   **AUROC**: 0.96  
*   **Clustering Accuracy**: 0.89
*   **Speedup**: 5-20× vs sampling-based methods
*   **Single Forward Pass**: Real-time detection

## 🏗️ Architecture Overview

LSD analyzes semantic trajectories through four key stages:

1.  **Hidden State Extraction**: Extract layer-wise activations from transformer models
2.  **Semantic Alignment**: Project hidden states and truth embeddings to shared space
3.  **Trajectory Analysis**: Compute geometric metrics (alignment, velocity, acceleration)
4.  **Risk Assessment**: Statistical validation and confidence scoring

## 🛠️ Core Components

### Configuration

```python
from lsd_framework import LayerwiseSemanticDynamicsConfig, OperationMode

config = LayerwiseSemanticDynamicsConfig(
    model_name="gpt2",
    truth_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
    num_pairs=1000,
    mode=OperationMode.HYBRID,
    batch_size=8,
    learning_rate=5e-5,
    shared_dim=256
)
```

### Training Projection Heads

```python
from lsd_framework import train_projection_heads, DataManager

# Build dataset
data_manager = DataManager(config)
pairs = data_manager.build_dataset()

# Train projection networks
model_manager = train_projection_heads(config, pairs)
```

### Feature Extraction

```python
from lsd_framework import FeatureExtractor

# Extract trajectory features
feature_extractor = FeatureExtractor(model_manager)
features = feature_extractor.extract_trajectory_features(
    text="The Earth orbits the Sun.",
    truth="Earth revolves around the Sun once a year."
)

print(features)
# {
#   'final_alignment': 0.855,
#   'mean_alignment': 0.598, 
#   'max_alignment': 0.912,
#   'convergence_layer': 8,
#   'stability': 0.069,
#   'alignment_gain': 0.198,
#   'mean_velocity': 0.263,
#   'mean_acceleration': 0.342,
#   'oscillation_count': 2
# }
```

### Comprehensive Analysis

```python
from lsd_framework import AnalysisOrchestrator

# Run complete analysis pipeline
orchestrator = AnalysisOrchestrator(config)
final_report = orchestrator.run_comprehensive_analysis()

# Access results
print(f"Best F1 Score: {final_report['key_findings']['best_f1_score']:.3f}")
print(f"Detection Quality: {final_report['key_findings']['detection_quality']}")
```

## 📈 Key Features

### 🎯 Multi-Modal Evaluation
*   **Supervised**: Logistic Regression, Random Forest, Gradient Boosting
*   **Unsupervised**: K-means clustering, Gaussian Mixture Models  
*   **Hybrid**: Combined supervised + unsupervised scoring

### 📊 Comprehensive Metrics
*   **Alignment Metrics**: Final, mean, and maximum layer alignment
*   **Dynamics Metrics**: Semantic velocity and acceleration
*   **Convergence Metrics**: Stability and oscillation patterns
*   **Statistical Validation**: Effect sizes and significance testing

### 🔧 Advanced Capabilities
*   **Model Agnostic**: Works with any transformer architecture
*   **Real-time Detection**: Single forward pass inference
*   **Interpretable Results**: Geometric trajectory visualization
*   **Confidence Calibration**: Probabilistic risk scoring

## 🎪 Operation Modes

### Supervised Mode
```python
config = LayerwiseSemanticDynamicsConfig(mode=OperationMode.SUPERVISED)
```
*Uses labeled data to train classifiers for hallucination detection*

### Unsupervised Mode  
```python
config = LayerwiseSemanticDynamicsConfig(mode=OperationMode.UNSUPERVISED)
```
*Uses clustering and anomaly detection without labels*

### Hybrid Mode
```python
config = LayerwiseSemanticDynamicsConfig(mode=OperationMode.HYBRID)
```
*Combines supervised and unsupervised approaches for robust detection*

## 📁 Project Structure

```
layerwise_semantic_dynamics_system/
├── models/                 # Trained projection networks
├── plots/                  # Analysis visualizations  
├── results/               # Evaluation results and reports
├── data/                  # Dataset storage
├── cache/                 # Model caching
└── execution.log          # System logs
```

## 🔬 Advanced Usage

### Custom Dataset Integration

```python
# Add custom factual-hallucination pairs
custom_pairs = [
    ("The capital of France is Paris.", "Paris is the capital city of France.", "factual"),
    ("The capital of France is London.", "Paris is the capital city of France.", "hallucination"),
]

data_manager = DataManager(config)
all_pairs = data_manager.build_dataset() + custom_pairs
```

### Model Scaling

```python
# Use larger models for improved performance
config = LayerwiseSemanticDynamicsConfig(
    model_name="gpt2-large",
    truth_encoder_name="sentence-transformers/all-mpnet-base-v2",
    shared_dim=512
)
```

### Batch Processing

```python
from lsd_framework import analyze_layerwise_dynamics

# Analyze multiple samples
df, trajectories, layerwise_data = analyze_layerwise_dynamics(
    pairs, 
    model_manager
)

# Export results
df.to_csv("trajectory_analysis.csv", index=False)
```

## 📚 Citation

If you use LSD in your research, please cite our paper:

```bibtex
@article{mir2024geometry,
  title={The Geometry of Truth: Layer-wise Semantic Dynamics for Hallucination Detection in Large Language Models},
  author={Mir, Amir Hameed},
  journal={arXiv preprint},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

*   📖 [Documentation](docs/)
*   🐛 [Issue Tracker](https://github.com/sirraya-tech/Sirraya_LSD_Code/issues)
*   💬 [Discussions](https://github.com/sirraya-tech/Sirraya_LSD_Code/discussions)

---

<p align="center">
  <em>Unveiling the Geometry of Truth in Language Models</em>
</p>
```