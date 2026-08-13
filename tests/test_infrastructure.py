import os
import sys
import tempfile
import numpy as np
import torch
import pytest

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.seed import set_seed
from src.utils.logging_config import setup_logger
from src.utils.config_loader import load_yaml_config, load_all_configs

def test_set_seed_determinism():
    """Verify that setting the global seed produces 100% deterministic random numbers."""
    set_seed(42)
    rand_a = np.random.rand(5)
    torch_a = torch.randn(5)

    set_seed(42)
    rand_b = np.random.rand(5)
    torch_b = torch.randn(5)

    assert np.allclose(rand_a, rand_b), "NumPy random seed is non-deterministic!"
    assert torch.allclose(torch_a, torch_b), "PyTorch random seed is non-deterministic!"

def test_config_loader():
    """Verify loading of YAML configurations."""
    configs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    all_configs = load_all_configs(configs_dir)

    assert "config" in all_configs
    assert "dataset" in all_configs
    assert "federated" in all_configs
    assert "continual" in all_configs
    assert "experiments" in all_configs

    assert all_configs["config"]["system"]["seed"] == 42
    assert all_configs["federated"]["federated"]["num_clients"] == 5
    assert len(all_configs["federated"]["federated"]["simulated_hospitals"]) == 5

def test_logger_creation():
    """Verify logging setup without errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test.log")
        logger = setup_logger(name="test_logger", log_file=log_file)
        logger.info("Testing infrastructure log entry.")

        assert os.path.exists(log_file)
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Testing infrastructure log entry." in content

        # Close handlers so Windows temporary directory cleanup can remove the log file
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

