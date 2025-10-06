#!/usr/bin/env python3
"""
Script to generate folder structure for Sirraya LSD Code
"""

import os
from pathlib import Path

def create_structure():
    """Create the complete folder structure"""
    
    # Define the directory structure
    directories = [
        ".github/workflows",
        ".github/ISSUE_TEMPLATE",
        "docs",
        "examples", 
        "lsd/core",
        "lsd/models",
        "lsd/data",
        "lsd/evaluation",
        "lsd/visualization",
        "tests",
        "scripts",
        "requirements"
    ]
    
    # Create directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}/")
    
    # Create empty __init__.py files
    init_files = [
        "lsd/__init__.py",
        "lsd/core/__init__.py", 
        "lsd/models/__init__.py",
        "lsd/data/__init__.py",
        "lsd/evaluation/__init__.py",
        "lsd/visualization/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"Created: {init_file}")
    
    # Create other essential files
    essential_files = [
        "README.md",
        "LICENSE",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "pyproject.toml",
        "requirements.txt",
        "environment.yml",
        "Dockerfile",
        ".dockerignore",
        ".gitignore",
        "setup.py",
        ".github/workflows/python-package.yml",
        ".github/workflows/tests.yml",
        ".github/ISSUE_TEMPLATE/bug_report.md",
        ".github/ISSUE_TEMPLATE/feature_request.md",
        "examples/basic_usage.py",
        "examples/advanced_analysis.py",
        "lsd/cli.py",
        "lsd/_version.py"
    ]
    
    for file in essential_files:
        Path(file).touch()
        print(f"Created: {file}")
    
    print("\nFolder structure created successfully!")
    print("Now you can add your code files to the respective directories.")

if __name__ == "__main__":
    create_structure()