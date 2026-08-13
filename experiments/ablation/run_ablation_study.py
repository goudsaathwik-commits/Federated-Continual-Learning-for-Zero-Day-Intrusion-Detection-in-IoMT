import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.evaluation.ablation_study import AblationStudyEngine

if __name__ == "__main__":
    print("Executing Phase 15: Component Ablation & Sensitivity Study (A1 - A8)...")
    engine = AblationStudyEngine(data_dir="data/processed", ablation_dir="results/ablation")
    results = engine.run_ablation_suite(seed=42)

    print("\nAblation Study Completed Successfully!")
    print("Results saved under: results/ablation/")
