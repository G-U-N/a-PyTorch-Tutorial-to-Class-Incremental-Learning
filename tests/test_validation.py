"""Validation tests to ensure the testing infrastructure works correctly."""

import pytest
import torch
import numpy as np
from pathlib import Path


def test_pytest_is_working():
    """Basic test to verify pytest is functioning."""
    assert True


def test_torch_import():
    """Test that PyTorch can be imported and basic operations work."""
    tensor = torch.tensor([1, 2, 3, 4])
    assert tensor.sum().item() == 10


def test_numpy_import():
    """Test that NumPy can be imported and basic operations work."""
    array = np.array([1, 2, 3, 4])
    assert array.sum() == 10


def test_fixtures_work(temp_dir, mock_config, sample_tensor):
    """Test that custom fixtures are working properly."""
    assert temp_dir.exists()
    assert isinstance(mock_config, dict)
    assert "model" in mock_config
    assert sample_tensor.shape == (4, 3, 32, 32)


@pytest.mark.unit
def test_unit_marker():
    """Test the unit marker is working."""
    assert True


@pytest.mark.integration
def test_integration_marker():
    """Test the integration marker is working."""
    assert True


@pytest.mark.slow
def test_slow_marker():
    """Test the slow marker is working."""
    assert True


def test_coverage_reporting():
    """Test that will show up in coverage reports."""
    def covered_function():
        return "covered"
    
    result = covered_function()
    assert result == "covered"