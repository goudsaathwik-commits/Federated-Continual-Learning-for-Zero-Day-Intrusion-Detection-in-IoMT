import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.experiment_runner import MasterExperimentRunner

def test_master_experiment_runner():
    """Verify MasterExperimentRunner executes benchmark suite and saves master_experiment_suite.json."""
    data_dir = "data/processed"
    if not os.path.exists(os.path.join(data_dir, "X_train.npy")):
        pytest.skip("Processed dataset arrays not present.")

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = MasterExperimentRunner(data_dir=data_dir, results_dir=tmpdir, models_dir=os.path.join(tmpdir, "models"))
        master_records = runner.run_master_benchmark(seeds=[42])

        assert "seed_42" in master_records
        assert "E1_Centralized" in master_records["seed_42"]
        assert "E3_FedAvg" in master_records["seed_42"]
        assert "E7_Proposed" in master_records["seed_42"]
        assert os.path.exists(os.path.join(tmpdir, "raw", "master_experiment_suite.json"))
