import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.experiment_runner import MasterExperimentRunner
from src.utils.seed import set_seed

if __name__ == "__main__":
    set_seed(42)
    print("Executing Phase 12: Master Experiment Suite (E1 - E7)...")
    runner = MasterExperimentRunner(data_dir="data/processed", results_dir="results", models_dir="models")
    master_records = runner.run_master_benchmark(seeds=[42])

    print("\nMaster Experiment Suite Completed Successfully!")
    print(f"Recorded results for seeds: {list(master_records.keys())}")
