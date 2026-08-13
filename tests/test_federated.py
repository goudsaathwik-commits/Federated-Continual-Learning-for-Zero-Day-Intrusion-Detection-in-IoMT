import os
import sys
import torch
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.federated.fedavg import aggregate_fedavg

def test_fedavg_exact_mathematical_weighting():
    """
    Mathematical Proof Test for FedAvg Weighting:
    Client 1: n1 = 100, weights = [1.0, 2.0]
    Client 2: n2 = 300, weights = [5.0, 6.0]
    Total N = 400
    Expected Global: (100/400)*[1, 2] + (300/400)*[5, 6] = [4.0, 5.0]
    """
    state_dict_1 = {"layer.weight": torch.tensor([1.0, 2.0])}
    state_dict_2 = {"layer.weight": torch.tensor([5.0, 6.0])}

    client_weights = [state_dict_1, state_dict_2]
    sample_counts = [100, 300]

    aggregated = aggregate_fedavg(client_weights, sample_counts)

    expected_tensor = torch.tensor([4.0, 5.0])
    assert torch.allclose(aggregated["layer.weight"], expected_tensor), f"FedAvg aggregation error! Got {aggregated['layer.weight']}, expected {expected_tensor}"

def test_fedavg_empty_input_handling():
    """Verify exception handling for empty client weights or invalid inputs."""
    with pytest.raises(ValueError):
        aggregate_fedavg([], [])

    with pytest.raises(ValueError):
        aggregate_fedavg([{"a": torch.tensor(1.0)}], [10, 20])
