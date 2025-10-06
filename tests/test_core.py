import pytest
from lsd.core import LayerwiseSemanticDynamicsConfig, OperationMode

def test_config_initialization():
    """Test configuration initialization with defaults"""
    config = LayerwiseSemanticDynamicsConfig()
    assert config.model_name == "gpt2"
    assert config.mode == OperationMode.HYBRID
    assert config.num_pairs == 1000

def test_config_custom():
    """Test configuration with custom values"""
    config = LayerwiseSemanticDynamicsConfig(
        model_name="gpt2-medium",
        num_pairs=500,
        mode=OperationMode.SUPERVISED
    )
    assert config.model_name == "gpt2-medium"
    assert config.num_pairs == 500
    assert config.mode == OperationMode.SUPERVISED