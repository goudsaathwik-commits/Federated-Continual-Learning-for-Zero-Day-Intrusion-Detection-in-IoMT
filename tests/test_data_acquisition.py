import os
import sys
import tempfile
import json
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import EdgeIIoTLoader
from src.data.validation import DatasetValidator

def test_dataset_loader_synthetic_generation():
    """Verify loader generates schema-compliant dataset when raw file is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = EdgeIIoTLoader(raw_dir=tmpdir, seed=42)
        df = loader.load_dataset(num_samples_fallback=100)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert "Attack_type" in df.columns
        assert "Attack_label" in df.columns
        assert "tcp.len" in df.columns

def test_dataset_validator():
    """Verify dataset validator computes missing values, duplicates, infs, and outputs profiles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = EdgeIIoTLoader(raw_dir=tmpdir, seed=42)
        df = loader.load_dataset(num_samples_fallback=200)

        results_dir = os.path.join(tmpdir, "results")
        validator = DatasetValidator(target_column="Attack_type")
        profile = validator.validate_and_profile(df, results_dir=results_dir)

        assert "dataset_summary" in profile
        assert profile["dataset_summary"]["num_samples"] == 200
        assert os.path.exists(os.path.join(results_dir, "dataset_profile.json"))
        assert os.path.exists(os.path.join(results_dir, "dataset_profile.csv"))
        assert os.path.exists(os.path.join(results_dir, "figures", "dataset_class_distribution.png"))

        # Read JSON back and verify keys
        with open(os.path.join(results_dir, "dataset_profile.json"), 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert "class_distribution_counts" in data
