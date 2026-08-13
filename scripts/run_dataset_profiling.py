import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import EdgeIIoTLoader
from src.data.validation import DatasetValidator
from src.utils.seed import set_seed

if __name__ == "__main__":
    set_seed(42)
    print("Loading Edge-IIoTset benchmark dataset...")
    loader = EdgeIIoTLoader(raw_dir="data/raw", seed=42)
    df = loader.load_dataset(num_samples_fallback=10000)

    print("Executing dataset validation and profiling...")
    validator = DatasetValidator(target_column="Attack_type")
    profile = validator.validate_and_profile(df, results_dir="results")

    print("Dataset profiling completed successfully!")
    print(f"Total Samples: {profile['dataset_summary']['num_samples']}")
    print(f"Total Features: {profile['dataset_summary']['num_features']}")
    print(f"Missing Values: {profile['dataset_summary']['missing_values_total']}")
    print(f"Duplicate Rows: {profile['dataset_summary']['duplicate_rows_total']}")
    print(f"Infinite Values: {profile['dataset_summary']['infinite_values_total']}")
