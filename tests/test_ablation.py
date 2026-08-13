import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.ablation_study import AblationStudyEngine

def test_ablation_study_engine():
    """Verify AblationStudyEngine executes A1-A5 and saves JSON and CSV tables."""
    data_dir = "data/processed"
    if not os.path.exists(os.path.join(data_dir, "X_train.npy")):
        pytest.skip("Processed dataset arrays not present.")

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = AblationStudyEngine(data_dir=data_dir, ablation_dir=tmpdir)
        results = engine.run_ablation_suite(seed=42)

        assert "A1_Centralized" in results
        assert "A2_FedAvg_No_CL" in results
        assert "A3_CL_No_FL" in results
        assert "A4_FL_CL_No_Replay" in results
        assert "A5_FL_CL_With_Replay" in results
        assert os.path.exists(os.path.join(tmpdir, "ablation_metrics.json"))
        assert os.path.exists(os.path.join(tmpdir, "ablation_comparison_table.csv"))
