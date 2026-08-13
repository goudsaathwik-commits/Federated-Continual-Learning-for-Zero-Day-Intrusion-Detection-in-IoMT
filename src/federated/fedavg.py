import copy
from typing import List, Dict
import torch

def aggregate_fedavg(client_weights: List[Dict[str, torch.Tensor]], sample_counts: List[int]) -> Dict[str, torch.Tensor]:
    r"""
    Computes weighted average of client neural network weights using the standard FedAvg formula:
    w_global = \sum_{k=1}^K (n_k / N_total) * w_k

    Args:
        client_weights: List of state_dict dictionaries from participating client models.
        sample_counts: List of local training sample counts n_k per client.
    Returns:
        Aggregated global state_dict.
    """
    if len(client_weights) == 0:
        raise ValueError("Cannot aggregate empty client weights list!")
    if len(client_weights) != len(sample_counts):
        raise ValueError(f"Mismatch between client weights count ({len(client_weights)}) and sample counts ({len(sample_counts)})")

    total_samples = sum(sample_counts)
    if total_samples <= 0:
        raise ValueError("Total sample count across clients must be > 0!")

    # Deep copy structure of first client to store aggregated weights
    aggregated_weights = copy.deepcopy(client_weights[0])

    # Zero out all parameter tensors in aggregated_weights before accumulating
    for key in aggregated_weights.keys():
        aggregated_weights[key] = torch.zeros_like(aggregated_weights[key], dtype=torch.float32)

    # Accumulate weighted parameters: w_global += (n_k / N_total) * w_k
    for k, client_state in enumerate(client_weights):
        weight_factor = sample_counts[k] / total_samples
        for key in aggregated_weights.keys():
            aggregated_weights[key] += client_state[key].to(torch.float32) * weight_factor

    return aggregated_weights
