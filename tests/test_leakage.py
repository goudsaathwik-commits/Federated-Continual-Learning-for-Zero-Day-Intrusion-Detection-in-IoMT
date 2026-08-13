import os
import sys
import json
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import EdgeIIoTLoader
from src.data.preprocessor import LeakageFreePreprocessor

def test_leakage_free_preprocessing_and_audit():
    """
    Automated Data Leakage Audit Test:
    1. Verifies zero index overlap between Train, Val, Test, and Zero-Day splits.
    2. Verifies scaler parameters match train split statistics ONLY.
    3. Verifies withheld zero-day classes never enter train, val, or test target sets.
    4. Produces results/leakage_audit.json.
    """
    # 1. Load benchmark dataset
    loader = EdgeIIoTLoader(seed=42)
    df = loader.load_dataset(num_samples_fallback=5000)

    # 2. Run Preprocessor
    preprocessor = LeakageFreePreprocessor(seed=42, withheld_classes=["Ransomware", "Backdoor"])
    cleaned_df = preprocessor.validate_and_clean_raw(df)
    train_df, val_df, test_df, zero_day_df = preprocessor.split_data(cleaned_df)

    # Audit 1: Disjoint Index Check
    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)
    zero_day_idx = set(zero_day_df.index)

    train_val_overlap = len(train_idx.intersection(val_idx))
    train_test_overlap = len(train_idx.intersection(test_idx))
    val_test_overlap = len(val_idx.intersection(test_idx))
    train_zeroday_overlap = len(train_idx.intersection(zero_day_idx))

    assert train_val_overlap == 0, f"DATA LEAKAGE DETECTED: {train_val_overlap} overlapping rows between train and val!"
    assert train_test_overlap == 0, f"DATA LEAKAGE DETECTED: {train_test_overlap} overlapping rows between train and test!"
    assert val_test_overlap == 0, f"DATA LEAKAGE DETECTED: {val_test_overlap} overlapping rows between val and test!"
    assert train_zeroday_overlap == 0, f"DATA LEAKAGE DETECTED: {train_zeroday_overlap} overlapping rows between train and zero-day!"

    # Fit preprocessor strictly on train_df
    preprocessor.fit_preprocessor(train_df)

    # Audit 2: Scaler Fitted strictly on Train
    num_cols = preprocessor.num_cols
    # Re-compute mean on train numerical features (imputed)
    train_num_imputed = preprocessor.num_imputer.transform(train_df[num_cols])
    computed_train_mean = np.mean(train_num_imputed, axis=0)

    # Compare fitted scaler mean with computed train mean
    scaler_fitted_on_train = np.allclose(preprocessor.scaler.mean_, computed_train_mean)
    assert scaler_fitted_on_train, "DATA LEAKAGE DETECTED: Scaler parameters do not match train split statistics!"

    # Audit 3: Zero-Day Target Isolation
    train_target_classes = set(train_df["Attack_type"].unique())
    val_target_classes = set(val_df["Attack_type"].unique())
    test_target_classes = set(test_df["Attack_type"].unique())

    zero_day_in_train = any(cls in train_target_classes for cls in preprocessor.withheld_classes)
    zero_day_in_val = any(cls in val_target_classes for cls in preprocessor.withheld_classes)
    zero_day_in_test = any(cls in test_target_classes for cls in preprocessor.withheld_classes)

    assert not zero_day_in_train, "DATA LEAKAGE DETECTED: Withheld zero-day attack present in train split!"
    assert not zero_day_in_val, "DATA LEAKAGE DETECTED: Withheld zero-day attack present in val split!"
    assert not zero_day_in_test, "DATA LEAKAGE DETECTED: Withheld zero-day attack present in test split!"

    # Transform all splits
    X_train, y_train = preprocessor.transform_data(train_df)
    X_val, y_val = preprocessor.transform_data(val_df)
    X_test, y_test = preprocessor.transform_data(test_df)
    X_zero_day, y_zero_day = preprocessor.transform_data(zero_day_df)

    # Save processed data
    preprocessor.save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, X_zero_day, y_zero_day, output_dir="data/processed")

    # Generate leakage audit report
    audit_report = {
        "audit_timestamp": "2026-08-12",
        "data_leakage_status": "PASSED_CLEAN",
        "leakage_detected": False,
        "disjointness_checks": {
            "train_val_overlap_count": train_val_overlap,
            "train_test_overlap_count": train_test_overlap,
            "val_test_overlap_count": val_test_overlap,
            "train_zeroday_overlap_count": train_zeroday_overlap,
            "is_disjoint": True
        },
        "preprocessor_fitting_audit": {
            "fitted_strictly_on_train": scaler_fitted_on_train,
            "train_sample_count_used_for_fit": len(train_df),
            "scaler_features_count": len(preprocessor.scaler.mean_)
        },
        "zero_day_isolation_audit": {
            "withheld_classes": preprocessor.withheld_classes,
            "present_in_train": zero_day_in_train,
            "present_in_val": zero_day_in_val,
            "present_in_test": zero_day_in_test,
            "is_completely_isolated": True
        }
    }

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    audit_file = os.path.join(results_dir, "leakage_audit.json")
    with open(audit_file, 'w', encoding='utf-8') as f:
        json.dump(audit_report, f, indent=2)

    assert os.path.exists(audit_file)
