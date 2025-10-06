# 🧠 Layer-wise Semantic Dynamics (LSD)
### *Geometric Hallucination Detection in Large Language Models*

[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Single-forward-pass hallucination detection through geometric analysis of semantic trajectories**

---

## 🎯 Overview

**LSD** is a novel framework that detects factual hallucinations in LLMs by analyzing how semantic representations evolve across transformer layers. Unlike traditional methods that require multiple sampling or external knowledge, LSD operates intrinsically with **single-pass efficiency** while achieving state-of-the-art performance.

### 🎪 Key Insight

```python
# Factual content converges toward truth embeddings
factual_trajectory    = "smooth → convergent → stable"

# Hallucinations diverge geometrically  
hallucination_trajectory = "oscillatory → divergent → unstable"