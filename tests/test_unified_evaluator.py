import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.unified_evaluator import UnifiedEvaluator

def test_security_metrics_calculation():
    """Verify UnifiedEvaluator calculates Attack Detection Rate, False Alarm Rate, and Zero-Day Detection Rate."""
    evaluator = UnifiedEvaluator()

    y_true = np.array([0, 0, 0, 1, 2, 3, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 2, 0, 0, 1])
    is_zero_day = np.array([0, 0, 0, 0, 0, 1, 0, 1]) # 2 zero-day attack samples

    metrics = evaluator.calculate_security_metrics(y_true, y_pred, is_zero_day_mask=is_zero_day)

    assert "attack_detection_rate" in metrics
    assert "false_alarm_rate" in metrics
    assert "zero_day_detection_rate" in metrics
    assert 0.0 <= metrics["attack_detection_rate"] <= 1.0
    assert 0.0 <= metrics["false_alarm_rate"] <= 1.0

def test_federated_metrics_calculation():
    """Verify UnifiedEvaluator computes FL rounds, cost, and participation rate."""
    evaluator = UnifiedEvaluator()
    history = [
        {"round": 1, "val_accuracy": 0.50, "cumulative_comm_mb": 2.1, "num_participating": 5},
        {"round": 2, "val_accuracy": 0.58, "cumulative_comm_mb": 4.2, "num_participating": 5},
        {"round": 3, "val_accuracy": 0.59, "cumulative_comm_mb": 6.3, "num_participating": 5}
    ]

    metrics = evaluator.calculate_federated_metrics(history)

    assert metrics["total_communication_rounds"] == 3
    assert metrics["cumulative_communication_mb"] == 6.3
    assert metrics["client_participation_rate_percent"] == 100.0

def test_generate_unified_report():
    """Verify UnifiedEvaluator generates unified summary JSON and CSV tables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = UnifiedEvaluator(results_dir=tmpdir)
        summary = evaluator.generate_unified_report()

        assert "evaluation_scope" in summary
        assert os.path.exists(os.path.join(tmpdir, "raw", "unified_evaluation_summary.json"))
        assert os.path.exists(os.path.join(tmpdir, "tables", "master_metrics_table.csv"))
