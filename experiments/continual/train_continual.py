import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.ids_backbone import TabularIDSBackbone
from src.continual.task_manager import TaskManager
from src.continual.cl_trainer import ContinualTrainer
from src.utils.seed import set_seed
from src.utils.logging_config import setup_logger

logger = setup_logger("train_continual")

def run_continual_learning_experiment():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running Centralized Continual Learning Baseline (E4) on device: {device}")

    # 1. Load preprocessed data
    data_dir = "data/processed"
    X_train = np.load(os.path.join(data_dir, "X_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(data_dir, "y_train.npy")).astype(np.int64)
    X_val = np.load(os.path.join(data_dir, "X_val.npy")).astype(np.float32)
    y_val = np.load(os.path.join(data_dir, "y_val.npy")).astype(np.int64)

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    # 2. Partition into 3 Sequential Tasks
    manager = TaskManager()
    task_train = manager.create_task_splits(X_train, y_train, num_tasks=3)
    task_val = manager.create_task_splits(X_val, y_val, num_tasks=3)

    model = TabularIDSBackbone(input_dim=input_dim, num_classes=num_classes, hidden_dims=[256, 128, 64], dropout=0.2)
    trainer = ContinualTrainer(model=model, device=device, num_classes=num_classes, seed=42)

    # 3. Strategy A: Naive Fine-Tuning (No Replay)
    logger.info("\n--- Running Strategy A: Sequential Fine-Tuning (Naive) ---")
    naive_results = trainer.train_sequential_tasks(
        task_train_data=task_train,
        task_val_data=task_val,
        use_replay=False,
        epochs_per_task=5,
        batch_size=64,
        lr=0.001
    )

    # 4. Strategy B: Continual Learning with Experience Replay
    logger.info("\n--- Running Strategy B: Continual Learning (Experience Replay) ---")
    replay_results = trainer.train_sequential_tasks(
        task_train_data=task_train,
        task_val_data=task_val,
        use_replay=True,
        replay_buffer_size=500,
        epochs_per_task=5,
        batch_size=64,
        lr=0.001
    )

    models_dir = "models/continual"
    raw_results_dir = "results/raw"
    tables_dir = "results/tables"
    figures_dir = "results/figures"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(raw_results_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Save Results JSON
    cl_all_results = {
        "naive_fine_tuning": naive_results,
        "experience_replay": replay_results
    }
    json_path = os.path.join(raw_results_dir, "continual_metrics.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(cl_all_results, f, indent=2)
    logger.info(f"Saved raw continual learning metrics JSON to: {json_path}")

    # Export Summary CSV
    summary_data = [
        {
            "Strategy": naive_results["strategy"],
            "Average_Accuracy": naive_results["average_accuracy"],
            "Backward_Transfer_BWT": naive_results["backward_transfer"],
            "Average_Forgetting": naive_results["average_forgetting"],
            "Task1_Final_Acc": naive_results["final_task_accuracies"][0],
            "Task2_Final_Acc": naive_results["final_task_accuracies"][1],
            "Task3_Final_Acc": naive_results["final_task_accuracies"][2]
        },
        {
            "Strategy": replay_results["strategy"],
            "Average_Accuracy": replay_results["average_accuracy"],
            "Backward_Transfer_BWT": replay_results["backward_transfer"],
            "Average_Forgetting": replay_results["average_forgetting"],
            "Task1_Final_Acc": replay_results["final_task_accuracies"][0],
            "Task2_Final_Acc": replay_results["final_task_accuracies"][1],
            "Task3_Final_Acc": replay_results["final_task_accuracies"][2]
        }
    ]
    csv_df = pd.DataFrame(summary_data)
    csv_path = os.path.join(tables_dir, "continual_results.csv")
    csv_df.to_csv(csv_path, index=False)
    logger.info(f"Saved continual learning summary CSV to: {csv_path}")

    # Generate Figures
    _plot_continual_learning_figures(naive_results, replay_results, figures_dir)

    logger.info("Continual Learning Baseline Experiment E4 completed successfully!")

def _plot_continual_learning_figures(naive_res: dict, replay_res: dict, figures_dir: str):
    """Generates Forgetting Curves and Accuracy Matrix Heatmaps."""
    sns.set_theme(style="whitegrid")

    # 1. Forgetting Curve / Task Retention Comparison
    tasks = ["Task 1", "Task 2", "Task 3"]
    plt.figure(figsize=(8, 5))
    plt.plot(tasks, naive_res["final_task_accuracies"], 'r--o', label="Naive Fine-Tuning", linewidth=2)
    plt.plot(tasks, replay_res["final_task_accuracies"], 'g-s', label="Experience Replay", linewidth=2)
    plt.title("Task Accuracy Retention (Mitigating Catastrophic Forgetting)", fontsize=12, fontweight='bold')
    plt.xlabel("Sequential Task Phase", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "continual_forgetting_curves.png"), dpi=300)
    plt.close()

    # 2. Task Accuracy Matrix Heatmap (Replay Strategy)
    plt.figure(figsize=(7, 5))
    R_mat = np.array(replay_res["R_matrix"])
    sns.heatmap(R_mat, annot=True, fmt=".2f", cmap="YlGn", xticklabels=tasks, yticklabels=[f"After {t}" for t in tasks])
    plt.title("Experience Replay Task Accuracy Matrix R_{i, j}", fontsize=12, fontweight='bold')
    plt.xlabel("Evaluated Task j", fontsize=11)
    plt.ylabel("Learned Task Phase i", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "continual_task_accuracy_matrix.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    run_continual_learning_experiment()
