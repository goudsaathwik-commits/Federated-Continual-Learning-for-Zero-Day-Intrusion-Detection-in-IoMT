import os
import sys
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.ids_backbone import TabularIDSBackbone
from src.zero_day.open_set_detector import EnergyBasedZeroDayDetector

def test_zero_day_data_isolation():
    """Verify zero-day withheld classes never exist in processed training or validation arrays."""
    data_dir = "data/processed"
    if not os.path.exists(os.path.join(data_dir, "y_train.npy")):
        pytest.skip("Processed dataset arrays not present.")

    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    y_zero_day = np.load(os.path.join(data_dir, "y_zero_day.npy"))

    # -1 represents withheld zero-day class in target encoding
    assert -1 not in y_train, "ZERO-DAY DATA LEAKAGE DETECTED in y_train!"
    assert -1 not in y_val, "ZERO-DAY DATA LEAKAGE DETECTED in y_val!"
    assert np.all(y_zero_day == -1), "y_zero_day array contains non-zero-day classes!"

def test_energy_based_zero_day_detector():
    """Verify EnergyBasedZeroDayDetector threshold fitting and open-set score computation."""
    device = torch.device("cpu")
    model = TabularIDSBackbone(input_dim=10, num_classes=5)

    X_val_known = np.random.randn(100, 10)
    X_test_known = np.random.randn(50, 10)
    X_test_zero_day = np.random.randn(50, 10) + 3.0 # Shifted OOD distribution

    detector = EnergyBasedZeroDayDetector(model=model, threshold_percentile=90.0)
    detector.fit_threshold(X_val_known, device=device)

    assert detector.fitted
    assert detector.threshold != 0.0

    eval_results = detector.evaluate_zero_day_detection(X_test_known, X_test_zero_day, device=device)

    expected_keys = ["threshold_tau", "zero_day_precision", "zero_day_recall", "zero_day_f1", "false_positive_rate", "false_negative_rate", "roc_auc", "pr_auc"]
    for key in expected_keys:
        assert key in eval_results, f"Missing zero-day evaluation key: {key}"
