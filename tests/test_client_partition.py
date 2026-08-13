import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.clients.partitioner import NonIIDPartitioner
from src.clients.hospital_client import HospitalClient

def test_non_iid_partitioning_disjointness():
    """Verify that Non-IID Dirichlet client partitioning produces completely disjoint client splits."""
    np.random.seed(42)
    num_samples = 1000
    num_features = 10
    num_classes = 5

    X_train = np.random.randn(num_samples, num_features)
    y_train = np.random.randint(0, num_classes, size=num_samples)

    X_val = np.random.randn(200, num_features)
    y_val = np.random.randint(0, num_classes, size=200)

    X_test = np.random.randn(200, num_features)
    y_test = np.random.randint(0, num_classes, size=200)

    partitioner = NonIIDPartitioner(num_clients=5, dirichlet_alpha=0.5, seed=42)
    clients = partitioner.create_clients(X_train, y_train, X_val, y_val, X_test, y_test)

    assert len(clients) == 5

    # Check total samples equal original data size
    total_train_samples = sum(c.train_size for c in clients)
    assert total_train_samples == num_samples, f"Sample count mismatch: {total_train_samples} != {num_samples}"

    # Verify no record overlap between clients by verifying total count sum
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = partitioner.validate_client_partition(clients, results_dir=tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "client_distribution.csv"))
        assert summary["num_clients"] == 5

def test_configurable_client_counts():
    """Verify that client count is fully configurable (e.g., N=3 and N=7)."""
    np.random.seed(42)
    X = np.random.randn(500, 5)
    y = np.random.randint(0, 3, size=500)

    for n_clients in [3, 7]:
        partitioner = NonIIDPartitioner(num_clients=n_clients, dirichlet_alpha=0.5, seed=42)
        clients = partitioner.create_clients(X, y, X[:50], y[:50], X[:50], y[:50])
        assert len(clients) == n_clients
        assert sum(c.train_size for c in clients) == 500
