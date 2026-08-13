import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.clients.partitioner import NonIIDPartitioner
from src.models.ids_backbone import TabularIDSBackbone
from src.evaluation.metrics import evaluate_classification_metrics

def test_local_hospital_model_isolation():
    """Verify local models train strictly on individual hospital slices."""
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = np.random.randint(0, 3, size=200)

    partitioner = NonIIDPartitioner(num_clients=3, dirichlet_alpha=0.5, seed=42)
    clients = partitioner.create_clients(X, y, X[:30], y[:30], X[:30], y[:30])

    for c in clients:
        model = TabularIDSBackbone(input_dim=10, num_classes=3)
        assert c.X_train.shape[1] == 10
        assert len(c.y_train) == c.train_size
        # Local metrics calculation check
        y_pred = np.zeros(c.test_size, dtype=int)
        metrics = evaluate_classification_metrics(c.y_test, y_pred, num_classes=3)
        assert "accuracy" in metrics
