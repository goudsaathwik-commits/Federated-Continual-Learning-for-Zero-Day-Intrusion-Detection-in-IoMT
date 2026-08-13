import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.unified_evaluator import UnifiedEvaluator

if __name__ == "__main__":
    print("Executing Phase 13: Unified Evaluation System...")
    evaluator = UnifiedEvaluator(results_dir="results")
    summary = evaluator.generate_unified_report()
    print("\nPhase 13 Evaluation Successfully Completed!")
    print("Exported: results/raw/unified_evaluation_summary.json & results/tables/master_metrics_table.csv")
