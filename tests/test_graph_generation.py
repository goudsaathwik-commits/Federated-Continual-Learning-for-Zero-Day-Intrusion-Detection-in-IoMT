import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_all_publication_graphs_exist():
    """Verify all 18 publication-quality figures exist in results/figures and are non-empty."""
    fig_dir = "results/figures"
    expected_figures = [
        "dataset_class_distribution.png",
        "attack_distribution.png",
        "hospital_sample_counts.png",
        "client_class_heatmap.png",
        "centralized_training_loss.png",
        "centralized_val_loss.png",
        "federated_accuracy_vs_round.png",
        "federated_f1_vs_round.png",
        "confusion_matrix_centralized.png",
        "roc_curves_comparison.png",
        "pr_curves_comparison.png",
        "per_class_f1_performance.png",
        "comparison_centralized_local_fl.png",
        "comparison_fl_vs_fl_cl.png",
        "continual_forgetting_curves.png",
        "zero_day_score_distribution.png",
        "federated_communication_cost.png",
        "ablation_components_f1.png"
    ]

    for fig_name in expected_figures:
        fig_path = os.path.join(fig_dir, fig_name)
        assert os.path.exists(fig_path), f"Missing publication figure: {fig_path}"
        assert os.path.getsize(fig_path) > 0, f"Figure file is 0 bytes: {fig_path}"
