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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.clients.partitioner import NonIIDPartitioner
from src.models.ids_backbone import TabularIDSBackbone
from src.continual.task_manager import TaskManager
from src.models.proposed_fcl_ids import ProposedFederatedContinualZeroDayIDS
from src.utils.seed import set_seed
from src.utils.logging_config import setup_logger

logger = setup_logger("run_full_benchmark")

def run_proposed_experiment():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Executing Proposed Federated Continual Zero-Day IDS Benchmark (E5 & E7) on device: {device}")

    # 1. Load preprocessed datasets
    data_dir = "data/processed"
    X_train = np.load(os.path.join(data_dir, "X_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(data_dir, "y_train.npy")).astype(np.int64)
    X_val = np.load(os.path.join(data_dir, "X_val.npy")).astype(np.float32)
    y_val = np.load(os.path.join(data_dir, "y_val.npy")).astype(np.int64)
    X_test = np.load(os.path.join(data_dir, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(data_dir, "y_test.npy")).astype(np.int64)
    X_zero_day = np.load(os.path.join(data_dir, "X_zero_day.npy")).astype(np.float32)

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    # 2. Partition into 5 Hospital Clients under Dirichlet non-IID skew (alpha=0.5)
    partitioner = NonIIDPartitioner(num_clients=5, dirichlet_alpha=0.5, seed=42)
    clients = partitioner.create_clients(X_train, y_train, X_val, y_val, X_test, y_test)

    # 3. Create 3 Continual Tasks per client & for global val set
    task_manager = TaskManager()
    
    # Per-client task splits
    # task_train_splits[t_idx][client_idx] -> (X, y)
    num_tasks = 3
    task_train_splits = [[] for _ in range(num_tasks)]

    for client in clients:
        c_task_splits = task_manager.create_task_splits(client.X_train, client.y_train, num_tasks=num_tasks)
        for t_idx in range(num_tasks):
            task_train_splits[t_idx].append(c_task_splits[t_idx])

    global_task_val_splits = task_manager.create_task_splits(X_val, y_val, num_tasks=num_tasks)

    # 4. Instantiate Proposed Unified Engine
    global_model = TabularIDSBackbone(input_dim=input_dim, num_classes=num_classes, hidden_dims=[256, 128, 64], dropout=0.2)
    engine = ProposedFederatedContinualZeroDayIDS(global_model=global_model, clients=clients, device=device, replay_buffer_size=500, seed=42)

    models_dir = "models/proposed"
    raw_results_dir = "results/raw"
    tables_dir = "results/tables"
    figures_dir = "results/figures"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(raw_results_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 5. Execute Proposed Pipeline
    results = engine.run_proposed_pipeline(
        task_train_splits=task_train_splits,
        task_val_splits=global_task_val_splits,
        X_zero_day_test=X_zero_day,
        num_fl_rounds_per_task=3,
        local_epochs=2,
        batch_size=64,
        lr=0.001,
        num_classes=num_classes
    )

    # Save final model checkpoint
    final_model_path = os.path.join(models_dir, "proposed_fcl_ids.pt")
    torch.save(engine.global_model.state_dict(), final_model_path)
    logger.info(f"Saved final Proposed FCL model checkpoint to: {final_model_path}")

    # Save Metrics JSON
    json_path = os.path.join(raw_results_dir, "proposed_fcl_metrics.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved proposed FCL metrics JSON to: {json_path}")

    # Save Results CSV Table
    csv_data = [{
        "Method": "Proposed FL + CL + Open-Set Zero-Day IDS",
        "Average_Accuracy": results["average_accuracy"],
        "Backward_Transfer_BWT": results["backward_transfer_bwt"],
        "Task1_Final_Acc": results["final_task_accuracies"][0],
        "Task2_Final_Acc": results["final_task_accuracies"][1],
        "Task3_Final_Acc": results["final_task_accuracies"][2],
        "Zero_Day_ROC_AUC_Final": results["zero_day_evaluations_per_task"][-1]["roc_auc"],
        "Zero_Day_Precision_Final": results["zero_day_evaluations_per_task"][-1]["zero_day_precision"]
    }]
    csv_df = pd.DataFrame(csv_data)
    csv_path = os.path.join(tables_dir, "proposed_fcl_results.csv")
    csv_df.to_csv(csv_path, index=False)
    logger.info(f"Saved proposed results summary CSV to: {csv_path}")

    # Generate Figures
    _plot_proposed_figures(results, figures_dir)

    logger.info(f"Proposed Framework Benchmark E5/E7 Completed! Average Accuracy: {results['average_accuracy']:.4f} | BWT: {results['backward_transfer_bwt']:.4f} | Zero-Day ROC-AUC: {results['zero_day_evaluations_per_task'][-1]['roc_auc']:.4f}")

def _plot_proposed_figures(results: dict, figures_dir: str):
    """Plots task accuracy matrix heatmap, forgetting curve, and zero-day AUROC trends."""
    sns.set_theme(style="whitegrid")
    tasks = ["Task 1", "Task 2", "Task 3"]

    # 1. Proposed Task Accuracy Heatmap
    plt.figure(figsize=(7, 5))
    R_mat = np.array(results["proposed_R_matrix"])
    sns.heatmap(R_mat, annot=True, fmt=".2f", cmap="Blues", xticklabels=tasks, yticklabels=[f"After {t}" for t in tasks])
    plt.title("Proposed Framework Task Accuracy Matrix R_{i, j}", fontsize=12, fontweight='bold')
    plt.xlabel("Evaluated Task j", fontsize=11)
    plt.ylabel("Learned Task Phase i", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "proposed_task_accuracy_matrix.png"), dpi=300)
    plt.close()

    # 2. Forgetting Curve
    plt.figure(figsize=(8, 5))
    plt.plot(tasks, results["final_task_accuracies"], 'b-o', label="Proposed FL + CL", linewidth=2)
    plt.title("Proposed Framework Task Accuracy Retention", fontsize=12, fontweight='bold')
    plt.xlabel("Sequential Task Phase", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "proposed_forgetting_curve.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    run_proposed_experiment()
