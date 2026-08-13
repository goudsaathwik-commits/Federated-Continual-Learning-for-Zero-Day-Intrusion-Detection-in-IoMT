import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.clients.partitioner import NonIIDPartitioner
from src.models.ids_backbone import TabularIDSBackbone
from src.federated.server import FederatedServer
from src.evaluation.metrics import evaluate_classification_metrics
from src.utils.seed import set_seed
from src.utils.logging_config import setup_logger

logger = setup_logger("train_fedavg")

def run_fedavg_experiment():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running Standard FedAvg Federated Learning Experiment (E3) on device: {device}")

    # 1. Load processed data
    data_dir = "data/processed"
    X_train = np.load(os.path.join(data_dir, "X_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(data_dir, "y_train.npy")).astype(np.int64)
    X_val = np.load(os.path.join(data_dir, "X_val.npy")).astype(np.float32)
    y_val = np.load(os.path.join(data_dir, "y_val.npy")).astype(np.int64)
    X_test = np.load(os.path.join(data_dir, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(data_dir, "y_test.npy")).astype(np.int64)

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    # 2. Partition into 5 Non-IID Hospital Clients (alpha=0.5)
    partitioner = NonIIDPartitioner(num_clients=5, dirichlet_alpha=0.5, seed=42)
    clients = partitioner.create_clients(X_train, y_train, X_val, y_val, X_test, y_test)

    # 3. Instantiate Global Model & Federated Server
    global_model = TabularIDSBackbone(input_dim=input_dim, num_classes=num_classes, hidden_dims=[256, 128, 64], dropout=0.2)
    server = FederatedServer(global_model=global_model, clients=clients, device=device, seed=42)

    # 4. Run FedAvg Rounds
    num_rounds = 10
    local_epochs = 3
    batch_size = 64
    lr = 0.001

    history = server.run_federated_rounds(
        num_rounds=num_rounds,
        fraction_fit=1.0,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=lr,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        num_classes=num_classes
    )

    # Final Evaluation on Global Test Set
    final_test_metrics = server.evaluate_global_model(X_test, y_test, num_classes=num_classes, batch_size=batch_size)

    models_dir = "models/federated"
    raw_results_dir = "results/raw"
    tables_dir = "results/tables"
    figures_dir = "results/figures"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(raw_results_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Save Global Model Checkpoint
    model_path = os.path.join(models_dir, "fedavg_global_model.pt")
    torch.save(server.global_model.state_dict(), model_path)
    logger.info(f"Saved FedAvg global model checkpoint to: {model_path}")

    # Save Metrics JSON
    metrics_json_path = os.path.join(raw_results_dir, "federated_metrics.json")
    fed_results = {
        "final_test_metrics": final_test_metrics,
        "round_history": history
    }
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(fed_results, f, indent=2)
    logger.info(f"Saved federated metrics JSON to: {metrics_json_path}")

    # Export Round History CSV
    history_df = pd.DataFrame(history)
    csv_path = os.path.join(tables_dir, "federated_results.csv")
    history_df.to_csv(csv_path, index=False)
    logger.info(f"Saved federated results CSV to: {csv_path}")

    # Generate Figures
    _plot_federated_performance(history_df, figures_dir)

    logger.info(f"FedAvg Baseline E3 Final Test Accuracy: {final_test_metrics['accuracy']:.4f} | F1 Macro: {final_test_metrics['f1_macro']:.4f}")
    logger.info("Standard FedAvg Experiment E3 completed successfully!")

def _plot_federated_performance(df: pd.DataFrame, figures_dir: str):
    """Generates round-by-round accuracy, F1, and communication cost graphs."""
    sns.set_theme(style="whitegrid")

    # 1. Accuracy vs Round
    plt.figure(figsize=(8, 5))
    plt.plot(df["round"], df["val_accuracy"], 'b-o', label="Val Accuracy", linewidth=2)
    plt.plot(df["round"], df["test_accuracy"], 'g--s', label="Test Accuracy", linewidth=2)
    plt.title("FedAvg Global Model Accuracy vs Round", fontsize=12, fontweight='bold')
    plt.xlabel("Federated Communication Round", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "federated_accuracy_vs_round.png"), dpi=300)
    plt.close()

    # 2. F1-Score vs Round
    plt.figure(figsize=(8, 5))
    plt.plot(df["round"], df["val_f1_macro"], 'm-o', label="Val F1 (Macro)", linewidth=2)
    plt.plot(df["round"], df["test_f1_macro"], 'c--s', label="Test F1 (Macro)", linewidth=2)
    plt.title("FedAvg Global Model F1-Score (Macro) vs Round", fontsize=12, fontweight='bold')
    plt.xlabel("Federated Communication Round", fontsize=11)
    plt.ylabel("F1 Score (Macro)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "federated_f1_vs_round.png"), dpi=300)
    plt.close()

    # 3. Communication Overhead Cumulative MB
    plt.figure(figsize=(8, 5))
    plt.plot(df["round"], df["cumulative_comm_mb"], 'r-^', label="Cumulative Communication (MB)", linewidth=2)
    plt.title("Cumulative Network Communication Overhead per Round", fontsize=12, fontweight='bold')
    plt.xlabel("Federated Communication Round", fontsize=11)
    plt.ylabel("Data Transferred (MB)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "federated_communication_cost.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    run_fedavg_experiment()
