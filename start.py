#!/usr/bin/env python3
"""
Script to generate the complete folder structure and files for 
Layer-wise Semantic Dynamics project
"""

import os
import sys
from pathlib import Path

def create_file(path, content):
    """Create a file with the given content"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Created: {path}")

def main():
    """Main function to create the project structure"""
    
    # Define the project structure
    project_files = {
        # Root directory files
        "pyproject.toml": """[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm"]
build-backend = "setuptools.build_meta"

[project]
name = "layerwise-semantic-dynamics"
dynamic = ["version"]
description = "Layer-wise Semantic Dynamics for Hallucination Detection in Language Models"
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
readme = "README.md"
license = {text = "MIT"}
keywords = ["nlp", "hallucination-detection", "transformers", "semantic-analysis"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
requires-python = ">=3.8"
dependencies = [
    "torch>=1.9.0",
    "transformers>=4.20.0",
    "sentence-transformers>=2.2.0",
    "datasets>=2.0.0",
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "tqdm>=4.64.0",
    "scipy>=1.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=6.0",
    "pytest-cov>=2.0",
    "black>=22.0",
    "isort>=5.0",
    "flake8>=4.0",
    "pre-commit>=2.0",
]
docs = [
    "sphinx>=4.0",
    "sphinx-rtd-theme>=1.0",
    "myst-parser>=0.18",
]

[project.urls]
Homepage = "https://github.com/yourusername/layerwise-semantic-dynamics"
Documentation = "https://yourusername.github.io/layerwise-semantic-dynamics"
Repository = "https://github.com/yourusername/layerwise-semantic-dynamics"
Issues = "https://github.com/yourusername/layerwise-semantic-dynamics/issues"

[project.scripts]
lsd-analyze = "lsd.cli:main"

[tool.setuptools_scm]
write_to = "lsd/_version.py"
""",

        "requirements.txt": """torch>=1.9.0
transformers>=4.20.0
sentence-transformers>=2.2.0
datasets>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.64.0
scipy>=1.7.0
""",

        "environment.yml": """name: lsd-env
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pytorch>=1.9.0
  - torchvision
  - torchaudio
  - cudatoolkit=11.3
  - transformers>=4.20.0
  - sentence-transformers>=2.2.0
  - datasets>=2.0.0
  - numpy>=1.21.0
  - pandas>=1.3.0
  - scikit-learn>=1.0.0
  - matplotlib>=3.5.0
  - seaborn>=0.11.0
  - tqdm>=4.64.0
  - scipy>=1.7.0
  - pip
  - pip:
    - pre-commit
""",

        "Dockerfile": """FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy package
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create results directory
RUN mkdir -p /app/results

# Set entrypoint
ENTRYPOINT ["python", "-m", "lsd.cli"]
""",

        ".dockerignore": """__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis
.DS_Store
""",

        ".gitignore": """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Results and outputs
results/
outputs/
models/
plots/
data/cache/

# Logs
*.log
logs/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
""",

        "README.md": """# Layer-wise Semantic Dynamics

[![Python Package](https://github.com/yourusername/layerwise-semantic-dynamics/actions/workflows/python-package.yml/badge.svg)](https://github.com/yourusername/layerwise-semantic-dynamics/actions/workflows/python-package.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/layerwise-semantic-dynamics.svg)](https://pypi.org/project/layerwise-semantic-dynamics/)

A comprehensive framework for detecting hallucinations in language models through layer-wise semantic dynamics analysis.

## Features

- **Layer-wise Analysis**: Track semantic evolution across transformer layers
- **Multiple Detection Modes**: Supervised, unsupervised, and hybrid approaches
- **Comprehensive Evaluation**: Extensive metrics and statistical analysis
- **Easy Integration**: Works with HuggingFace transformers
- **Production Ready**: CLI, Docker, and Python API support

## Installation

### From PyPI
```bash
pip install layerwise-semantic-dynamics