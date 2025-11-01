"""
Pytest configuration and shared fixtures for MDITRE test suite.

This file provides common test configuration, fixtures, and utilities
used across all test modules.
"""

import os
import sys
import pytest
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def device():
    """Provide PyTorch device for testing (CPU or CUDA)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def use_gpu():
    """Check if GPU is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture(scope="function")
def random_seed():
    """Provide consistent random seed for reproducible tests."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed


@pytest.fixture(scope="function")
def sample_data():
    """Generate sample microbiome data for testing."""
    batch_size = 4
    num_features = 10
    num_timepoints = 5
    
    # Simulated abundance data
    x = torch.randn(batch_size, num_features, num_timepoints)
    x = torch.abs(x)  # Abundances are non-negative
    
    # Binary labels
    y = torch.randint(0, 2, (batch_size,))
    
    # Time points
    times = torch.linspace(0, 1, num_timepoints).unsqueeze(0).repeat(batch_size, 1)
    
    return {
        'abundances': x,
        'labels': y,
        'times': times,
        'batch_size': batch_size,
        'num_features': num_features,
        'num_timepoints': num_timepoints
    }


@pytest.fixture(scope="session")
def test_output_dir(tmp_path_factory):
    """Create temporary directory for test outputs."""
    output_dir = tmp_path_factory.mktemp("test_outputs")
    return output_dir


# Configure pytest warnings
def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "architecture: marks architecture tests"
    )
    config.addinivalue_line(
        "markers", "layer1: marks layer 1 tests"
    )
    config.addinivalue_line(
        "markers", "layer2: marks layer 2 tests"
    )
    config.addinivalue_line(
        "markers", "layer3: marks layer 3 tests"
    )
    config.addinivalue_line(
        "markers", "layer4: marks layer 4 tests"
    )
    config.addinivalue_line(
        "markers", "layer5: marks layer 5 tests"
    )
    config.addinivalue_line(
        "markers", "differentiability: marks differentiability tests"
    )
    config.addinivalue_line(
        "markers", "critical: marks critical tests"
    )
    config.addinivalue_line(
        "markers", "model: marks model tests"
    )
    config.addinivalue_line(
        "markers", "phylogenetic: marks phylogenetic tests"
    )
    config.addinivalue_line(
        "markers", "temporal: marks temporal tests"
    )
    config.addinivalue_line(
        "markers", "metrics: marks metrics tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "seeding: marks seeding and reproducibility tests"
    )
    config.addinivalue_line(
        "markers", "integrity: marks package integrity tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip GPU tests if CUDA unavailable."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
