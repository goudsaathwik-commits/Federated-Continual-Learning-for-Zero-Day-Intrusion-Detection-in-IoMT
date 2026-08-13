import os
import sys
import torch
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.ids_backbone import TabularIDSBackbone
from src.evaluation.metrics import evaluate_classification_metrics

def test_tabular_ids_backbone_forward_pass():
    """Verify PyTorch TabularIDSBackbone forward pass dimensions."""
    batch_size = 16
    input_dim = 47
    num_classes = 12

    model = TabularIDSBackbone(input_dim=input_dim, num_classes=num_classes)
    x = torch.randn(batch_size, input_dim)
    logits = model(x)
    probas = model.predict_proba(x)

    assert logits.shape == (batch_size, num_classes)
    assert probas.shape == (batch_size, num_classes)
    assert torch.allclose(probas.sum(dim=1), torch.ones(batch_size), atol=1e-5)

def test_evaluate_classification_metrics():
    """Verify metrics calculator output dictionary keys."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])
    y_proba = np.eye(3)[y_pred]

    metrics = evaluate_classification_metrics(y_true, y_pred, y_proba, num_classes=3)

    expected_keys = [
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "balanced_accuracy", "specificity_macro", "false_positive_rate",
        "false_negative_rate", "confusion_matrix", "roc_auc_macro", "pr_auc_macro"
    ]
    for key in expected_keys:
        assert key in metrics, f"Missing metric key: {key}"
