import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.clients.partitioner import NonIIDPartitioner
from src.models.ids_backbone import TabularIDSBackbone
from src.evaluation.metrics import evaluate_classification_metrics
from src.utils.seed import set_seed
from src.utils.logging_config import setup_logger

logger = setup_logger("train_local")

def run_local_baseline_experiments():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running Independent Local Hospital IDS experiments (E2) on device: {device}")

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

    models_dir = "models/local"
    raw_results_dir = "results/raw"
    tables_dir = "results/tables"
    figures_dir = "results/figures"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(raw_results_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    client_results = []
    all_metrics = {}

    epochs = 10
    batch_size = 64

    # 3. Train Independent Models per Hospital Client
    for client in clients:
        logger.info(f"\n--- Training Independent Local IDS for {client.client_id} ---")
        logger.info(f"Local Train Samples: {client.train_size} | Val: {client.val_size} | Test: {client.test_size}")

        model = TabularIDSBackbone(input_dim=input_dim, num_classes=num_classes, hidden_dims=[256, 128, 64], dropout=0.2).to(device)

        c_train_loader = DataLoader(TensorDataset(torch.tensor(client.X_train), torch.tensor(client.y_train)), batch_size=batch_size, shuffle=True)
        c_val_loader = DataLoader(TensorDataset(torch.tensor(client.X_val), torch.tensor(client.y_val)), batch_size=batch_size, shuffle=False)
        c_test_loader = DataLoader(TensorDataset(torch.tensor(client.X_test), torch.tensor(client.y_test)), batch_size=batch_size, shuffle=False)
        global_test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

        best_val_acc = 0.0
        model_path = os.path.join(models_dir, f"{client.client_id}_model.pt")

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for bx, by in c_train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(by)

            # Local Validation
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for bx, by in c_val_loader:
                    bx, by = bx.to(device), by.to(device)
                    logits = model(bx)
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == by).sum().item()
                    val_total += len(by)

            val_acc = val_correct / val_total if val_total > 0 else 0.0
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_path)

        # Load best local checkpoint
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        # Evaluate on Local Test Set
        local_preds, local_probas = [], []
        with torch.no_grad():
            for bx, _ in c_test_loader:
                bx = bx.to(device)
                logits = model(bx)
                probas = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                local_preds.extend(preds.cpu().numpy())
                local_probas.extend(probas.cpu().numpy())

        local_metrics = evaluate_classification_metrics(client.y_test, np.array(local_preds), np.array(local_probas), num_classes=num_classes)

        # Evaluate on Global Test Set (Generalization Gap)
        global_preds, global_probas = [], []
        with torch.no_grad():
            for bx, _ in global_test_loader:
                bx = bx.to(device)
                logits = model(bx)
                probas = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                global_preds.extend(preds.cpu().numpy())
                global_probas.extend(probas.cpu().numpy())

        global_metrics = evaluate_classification_metrics(y_test, np.array(global_preds), np.array(global_probas), num_classes=num_classes)

        logger.info(f"{client.client_id} -> Local Test Acc: {local_metrics['accuracy']:.4f} | Global Test Acc: {global_metrics['accuracy']:.4f} | Global F1 Macro: {global_metrics['f1_macro']:.4f}")

        all_metrics[client.client_id] = {
            "local_evaluation": local_metrics,
            "global_evaluation": global_metrics
        }

        client_results.append({
            "Hospital_Client": client.client_id,
            "Train_Samples": client.train_size,
            "Local_Test_Accuracy": local_metrics["accuracy"],
            "Local_Test_F1_Macro": local_metrics["f1_macro"],
            "Global_Test_Accuracy": global_metrics["accuracy"],
            "Global_Test_F1_Macro": global_metrics["f1_macro"],
            "Generalization_Gap_Acc": round(local_metrics["accuracy"] - global_metrics["accuracy"], 4)
        })

    # Save metrics JSON & CSV
    metrics_json_path = os.path.join(raw_results_dir, "local_metrics.json")
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Saved local metrics JSON to: {metrics_json_path}")

    results_df = pd.DataFrame(client_results)
    csv_path = os.path.join(tables_dir, "local_results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved local results summary table to: {csv_path}")

    # Generate Comparison Figures
    _plot_local_vs_global_comparison(results_df, os.path.join(figures_dir, "local_vs_global_f1_comparison.png"))

    logger.info("Independent Local Baseline Experiment E2 completed successfully!")

def _plot_local_vs_global_comparison(df: pd.DataFrame, output_path: str):
    """Plots Local vs Global Accuracy comparison bar chart."""
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    df_melted = df.melt(id_vars=["Hospital_Client"],
                        value_vars=["Local_Test_Accuracy", "Global_Test_Accuracy"],
                        var_name="Evaluation_Type", value_name="Accuracy")
    
    df_melted["Hospital"] = df_melted["Hospital_Client"].apply(lambda x: x.replace("Hospital_", "H").replace("_", "\n"))

    ax = sns.barplot(x="Hospital", y="Accuracy", hue="Evaluation_Type", data=df_melted, palette="Set2")
    plt.title("Local Hospital IDS Generalization Gap Under Non-IID Skew", fontsize=12, fontweight='bold')
    plt.xlabel("Hospital Client Node", fontsize=11)
    plt.ylabel("Accuracy Score", fontsize=11)
    plt.ylim(0, 1.0)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height / 2.),
                        ha='center', va='center', fontsize=9, color='black', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    run_local_baseline_experiments()
