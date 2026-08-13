import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.ids_backbone import TabularIDSBackbone
from src.zero_day.open_set_detector import EnergyBasedZeroDayDetector
from src.evaluation.visualization import plot_confusion_matrix
from src.utils.seed import set_seed
from src.utils.logging_config import setup_logger

logger = setup_logger("eval_zero_day")

def run_zero_day_detection_experiment():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running Zero-Day Open-Set Detection Experiment (E6) on device: {device}")

    # 1. Load processed dataset splits
    data_dir = "data/processed"
    X_train = np.load(os.path.join(data_dir, "X_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(data_dir, "y_train.npy")).astype(np.int64)
    X_val = np.load(os.path.join(data_dir, "X_val.npy")).astype(np.float32)
    X_test = np.load(os.path.join(data_dir, "X_test.npy")).astype(np.float32)
    X_zero_day = np.load(os.path.join(data_dir, "X_zero_day.npy")).astype(np.float32)

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    logger.info(f"Loaded Known Test: {X_test.shape} | Withheld Zero-Day Test (Ransomware/Backdoor): {X_zero_day.shape}")

    # 2. Load trained backbone model
    model_path = "models/centralized/pytorch_mlp_ids.pt"
    if not os.path.exists(model_path):
        model_path = "models/federated/fedavg_global_model.pt"

    model = TabularIDSBackbone(input_dim=input_dim, num_classes=num_classes, hidden_dims=[256, 128, 64], dropout=0.2)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.to(device)

    # 3. Instantiate & Fit Energy-Based Zero-Day Detector on Known Validation Set
    detector = EnergyBasedZeroDayDetector(model=model, temperature=1.0, threshold_percentile=95.0)
    detector.fit_threshold(X_val, device=device)

    # 4. Evaluate Zero-Day Anomaly Detection
    eval_metrics = detector.evaluate_zero_day_detection(X_known_test=X_test, X_zero_day_test=X_zero_day, device=device)

    models_dir = "models/zero_day"
    raw_results_dir = "results/raw"
    tables_dir = "results/tables"
    figures_dir = "results/figures"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(raw_results_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Save Detector State Checkpoint
    torch.save({"model_state": model.state_dict(), "threshold": detector.threshold}, os.path.join(models_dir, "open_set_detector.pt"))

    # Save Metrics JSON
    json_path = os.path.join(raw_results_dir, "zero_day_metrics.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(eval_metrics, f, indent=2)
    logger.info(f"Saved raw zero-day detection metrics JSON to: {json_path}")

    # Export Summary CSV
    results_summary = [{
        "Method": "Energy-Based Scoring",
        "Threshold_Tau": eval_metrics["threshold_tau"],
        "Zero_Day_Precision": eval_metrics["zero_day_precision"],
        "Zero_Day_Recall": eval_metrics["zero_day_recall"],
        "Zero_Day_F1": eval_metrics["zero_day_f1"],
        "False_Positive_Rate": eval_metrics["false_positive_rate"],
        "False_Negative_Rate": eval_metrics["false_negative_rate"],
        "ROC_AUC": eval_metrics["roc_auc"],
        "PR_AUC": eval_metrics["pr_auc"]
    }]
    results_df = pd.DataFrame(results_summary)
    csv_path = os.path.join(tables_dir, "zero_day_results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved zero-day results summary CSV to: {csv_path}")

    # Generate Figures
    _plot_zero_day_figures(detector, X_test, X_zero_day, eval_metrics, figures_dir, device)

    logger.info(f"Zero-Day Detection Experiment E6 Completed! Precision: {eval_metrics['zero_day_precision']:.4f} | Recall: {eval_metrics['zero_day_recall']:.4f} | F1: {eval_metrics['zero_day_f1']:.4f} | ROC-AUC: {eval_metrics['roc_auc']:.4f}")

def _plot_zero_day_figures(detector: EnergyBasedZeroDayDetector, X_test: np.ndarray, X_zero_day: np.ndarray, metrics: dict, figures_dir: str, device: torch.device):
    """Generates Energy Score Distributions, Threshold Analysis, Confusion Matrix, and ROC/PR curves."""
    sns.set_theme(style="whitegrid")

    known_energies = detector.compute_energy_scores(X_test, device)
    zero_day_energies = detector.compute_energy_scores(X_zero_day, device)

    # 1. Score Distribution Plot
    plt.figure(figsize=(9, 5))
    sns.histplot(known_energies, color="blue", label="Known Traffic Energy (In-Distribution)", kde=True, stat="density", bins=40, alpha=0.5)
    sns.histplot(zero_day_energies, color="red", label="Zero-Day Attack Energy (OOD Withheld)", kde=True, stat="density", bins=40, alpha=0.5)
    plt.axvline(detector.threshold, color="black", linestyle="--", linewidth=2, label=f"Energy Threshold tau ({detector.threshold:.2f})")
    plt.title("Energy Score Distribution: Known Traffic vs Zero-Day Attacks", fontsize=12, fontweight='bold')
    plt.xlabel("Energy Score E(x; w)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "zero_day_score_distribution.png"), dpi=300)
    plt.close()

    # 2. Confusion Matrix
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, ["Known Traffic", "Zero-Day Attack"], os.path.join(figures_dir, "zero_day_confusion_matrix.png"), title="Zero-Day Open-Set Detection Confusion Matrix")

    # 3. ROC & PR Curves
    y_eval = np.array([0] * len(X_test) + [1] * len(X_zero_day))
    energies_all = np.concatenate([known_energies, zero_day_energies])

    fpr, tpr, _ = roc_curve(y_eval, energies_all)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, 'b-', label=f"Energy Detector (AUC = {metrics['roc_auc']:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
    plt.title("Zero-Day Anomaly Detection ROC Curve", fontsize=12, fontweight='bold')
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "zero_day_roc_curve.png"), dpi=300)
    plt.close()

    p, r, _ = precision_recall_curve(y_eval, energies_all)
    plt.figure(figsize=(7, 5))
    plt.plot(r, p, 'g-', label=f"Energy Detector (AUC = {metrics['pr_auc']:.4f})", linewidth=2)
    plt.title("Zero-Day Anomaly Detection Precision-Recall Curve", fontsize=12, fontweight='bold')
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "zero_day_pr_curve.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    run_zero_day_detection_experiment()
