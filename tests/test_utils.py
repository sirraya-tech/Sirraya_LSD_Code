import pytest
import numpy as np
import pandas as pd
from lsd.utils.helpers import convert_to_json_serializable, safe_json_dump
import tempfile
import json

def test_json_serialization():
    """Test JSON serialization of various types"""
    # Test numpy types
    data = {
        'int_val': np.int64(42),
        'float_val': np.float64(3.14),
        'array': np.array([1, 2, 3]),
        'bool_val': np.bool_(True)
    }
    
    serialized = convert_to_json_serializable(data)
    
    assert isinstance(serialized['int_val'], int)
    assert isinstance(serialized['float_val'], float)
    assert isinstance(serialized['array'], list)
    assert isinstance(serialized['bool_val'], bool)

def test_safe_json_dump():
    """Test safe JSON dumping to file"""
    data = {
        'test_array': np.array([1, 2, 3]),
        'test_float': np.float64(1.23)
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        safe_json_dump(data, temp_path)
        
        # Verify file can be loaded
        with open(temp_path, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['test_array'] == [1, 2, 3]
        assert loaded['test_float'] == 1.23
        
    finally:
        import os
        os.unlink(temp_path)