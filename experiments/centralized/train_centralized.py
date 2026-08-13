import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.classical_ids import build_classical_ids
from src.models.ids_backbone import TabularIDSBackbone
from src.evaluation.metrics import evaluate_classification_metrics
from src.evaluation.visualization import (
    plot_confusion_matrix, plot_training_curves, plot_roc_pr_curves
)
from src.utils.seed import set_seed
from src.utils.logging_config import setup_logger

logger = setup_logger("train_centralized")

def run_centralized_experiments():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running Centralized IDS experiments on device: {device}")

    # 1. Load preprocessed data splits
    data_dir = "data/processed"
    X_train = np.load(os.path.join(data_dir, "X_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(data_dir, "y_train.npy")).astype(np.int64)
    X_val = np.load(os.path.join(data_dir, "X_val.npy")).astype(np.float32)
    y_val = np.load(os.path.join(data_dir, "y_val.npy")).astype(np.int64)
    X_test = np.load(os.path.join(data_dir, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(data_dir, "y_test.npy")).astype(np.int64)

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    logger.info(f"Loaded dataset - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}, Input Dim: {input_dim}, Classes: {num_classes}")

    models_dir = "models/centralized"
    raw_results_dir = "results/raw"
    tables_dir = "results/tables"
    figures_dir = "results/figures"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(raw_results_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # -------------------------------------------------------------
    # BASELINE 1A: Classical Random Forest IDS
    # -------------------------------------------------------------
    logger.info("--- Training Classical Baseline: Random Forest ---")
    rf_model = build_classical_ids("random_forest", seed=42)
    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)
    rf_metrics = evaluate_classification_metrics(y_test, rf_pred, rf_proba, num_classes=num_classes)

    joblib.dump(rf_model, os.path.join(models_dir, "random_forest.joblib"))
    logger.info(f"Random Forest Test Accuracy: {rf_metrics['accuracy']:.4f} | F1 Macro: {rf_metrics['f1_macro']:.4f}")

    # -------------------------------------------------------------
    # BASELINE 1B: Primary PyTorch Neural Network (Tabular IDS Backbone)
    # -------------------------------------------------------------
    logger.info("--- Training Primary Neural Baseline: PyTorch Tabular IDS Backbone ---")
    model = TabularIDSBackbone(input_dim=input_dim, num_classes=num_classes, hidden_dims=[256, 128, 64], dropout=0.2).to(device)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    epochs = 15
    best_val_acc = 0.0
    best_model_path = os.path.join(models_dir, "pytorch_mlp_ids.pt")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train += len(batch_y)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation Phase (Model Selection)
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                val_running_loss += loss.item() * len(batch_y)
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == batch_y).sum().item()
                total_val += len(batch_y)

        epoch_val_loss = val_running_loss / total_val
        epoch_val_acc = correct_val / total_val

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        logger.info(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

        # Model selection checkpointing
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)

    # Load best validation model checkpoint for final test evaluation
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    all_test_preds = []
    all_test_probas = []

    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probas = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_test_preds.extend(preds.cpu().numpy())
            all_test_probas.extend(probas.cpu().numpy())

    nn_test_preds = np.array(all_test_preds)
    nn_test_probas = np.array(all_test_probas)

    nn_metrics = evaluate_classification_metrics(y_test, nn_test_preds, nn_test_probas, num_classes=num_classes)
    logger.info(f"PyTorch Neural Network Test Accuracy: {nn_metrics['accuracy']:.4f} | F1 Macro: {nn_metrics['f1_macro']:.4f}")

    # -------------------------------------------------------------
    # SAVE METRICS & GENERATE FIGURES
    # -------------------------------------------------------------
    all_metrics = {
        "RandomForest_Centralized": rf_metrics,
        "PyTorch_MLP_Centralized": nn_metrics
    }

    metrics_json_path = os.path.join(raw_results_dir, "centralized_metrics.json")
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Saved raw centralized metrics JSON to: {metrics_json_path}")

    # Results Table CSV
    results_summary = [
        {
            "Model": "Random Forest (Centralized)",
            "Accuracy": rf_metrics["accuracy"],
            "Precision_Macro": rf_metrics["precision_macro"],
            "Recall_Macro": rf_metrics["recall_macro"],
            "F1_Macro": rf_metrics["f1_macro"],
            "F1_Weighted": rf_metrics["f1_weighted"],
            "Balanced_Accuracy": rf_metrics["balanced_accuracy"],
            "Specificity_Macro": rf_metrics["specificity_macro"],
            "FPR": rf_metrics["false_positive_rate"],
            "FNR": rf_metrics["false_negative_rate"],
            "ROC_AUC": rf_metrics.get("roc_auc_macro", 0.0),
            "PR_AUC": rf_metrics.get("pr_auc_macro", 0.0)
        },
        {
            "Model": "PyTorch MLP (Centralized)",
            "Accuracy": nn_metrics["accuracy"],
            "Precision_Macro": nn_metrics["precision_macro"],
            "Recall_Macro": nn_metrics["recall_macro"],
            "F1_Macro": nn_metrics["f1_macro"],
            "F1_Weighted": nn_metrics["f1_weighted"],
            "Balanced_Accuracy": nn_metrics["balanced_accuracy"],
            "Specificity_Macro": nn_metrics["specificity_macro"],
            "FPR": nn_metrics["false_positive_rate"],
            "FNR": nn_metrics["false_negative_rate"],
            "ROC_AUC": nn_metrics.get("roc_auc_macro", 0.0),
            "PR_AUC": nn_metrics.get("pr_auc_macro", 0.0)
        }
    ]

    results_df = pd.DataFrame(results_summary)
    csv_path = os.path.join(tables_dir, "centralized_results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved centralized results summary table to: {csv_path}")

    # Generate Figures
    class_names = [f"Class_{i}" for i in range(num_classes)]
    cm = np.array(nn_metrics["confusion_matrix"])

    plot_confusion_matrix(cm, class_names, os.path.join(figures_dir, "centralized_confusion_matrix.png"), title="Centralized PyTorch MLP Confusion Matrix")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, os.path.join(figures_dir, "centralized_training_curves.png"))
    plot_roc_pr_curves(y_test, nn_test_probas,
                       os.path.join(figures_dir, "centralized_roc_curve.png"),
                       os.path.join(figures_dir, "centralized_pr_curve.png"))

    logger.info("Centralized Baseline Experiment E1 completed successfully!")

if __name__ == "__main__":
    run_centralized_experiments()
