import os
import sys
import torch
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.ids_backbone import TabularIDSBackbone
from src.clients.partitioner import NonIIDPartitioner
from src.models.proposed_fcl_ids import ProposedFederatedContinualZeroDayIDS

def test_proposed_fcl_pipeline_execution():
    """Verify proposed unified FCL pipeline initializes and executes without error."""
    device = torch.device("cpu")
    num_samples = 200
    num_features = 10
    num_classes = 4

    X_train = np.random.randn(num_samples, num_features)
    y_train = np.random.randint(0, num_classes, size=num_samples)
    X_val = np.random.randn(50, num_features)
    y_val = np.random.randint(0, num_classes, size=50)

    partitioner = NonIIDPartitioner(num_clients=3, dirichlet_alpha=0.5, seed=42)
    clients = partitioner.create_clients(X_train, y_train, X_val, y_val, X_val, y_val)

    global_model = TabularIDSBackbone(input_dim=num_features, num_classes=num_classes)
    engine = ProposedFederatedContinualZeroDayIDS(global_model=global_model, clients=clients, device=device, seed=42)

    assert len(engine.client_buffers) == 3

    # Create dummy 2 task splits for 3 clients
    task1_client_train = [(c.X_train[:20], c.y_train[:20]) for c in clients]
    task2_client_train = [(c.X_train[20:40], c.y_train[20:40]) for c in clients]
    task_train_splits = [task1_client_train, task2_client_train]

    task_val_splits = [(X_val[:25], y_val[:25]), (X_val[25:], y_val[25:])]
    X_zero_day_test = np.random.randn(30, num_features) + 2.0

    results = engine.run_proposed_pipeline(
        task_train_splits=task_train_splits,
        task_val_splits=task_val_splits,
        X_zero_day_test=X_zero_day_test,
        num_fl_rounds_per_task=1,
        local_epochs=1,
        batch_size=16,
        num_classes=num_classes
    )

    assert "proposed_R_matrix" in results
    assert "average_accuracy" in results
    assert "backward_transfer_bwt" in results
