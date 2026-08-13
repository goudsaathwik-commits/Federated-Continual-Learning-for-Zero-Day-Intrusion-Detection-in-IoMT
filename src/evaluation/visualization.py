import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc

def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: str, title: str = "Confusion Matrix"):
    """Plots and saves a styled confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_training_curves(train_losses: list, val_losses: list, train_accs: list, val_accs: list, output_path: str):
    """Plots training and validation loss/accuracy curves over epochs."""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.set_theme(style="whitegrid")

    # Loss plot
    ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r--s', label='Validation Loss', linewidth=2)
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-o', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'g--s', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_roc_pr_curves(y_true: np.ndarray, y_proba: np.ndarray, output_roc_path: str, output_pr_path: str):
    """Plots multi-class ROC and Precision-Recall curves."""
    unique_classes = np.unique(y_true)
    num_classes = len(unique_classes)

    # 1. ROC Curves
    plt.figure(figsize=(9, 7))
    sns.set_theme(style="whitegrid")
    for cls in unique_classes:
        if cls < y_proba.shape[1]:
            y_binary = (y_true == cls).astype(int)
            fpr, tpr, _ = roc_curve(y_binary, y_proba[:, cls])
            roc_auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {cls} (AUC = {roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.title('Multi-Class Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold')
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_roc_path), exist_ok=True)
    plt.savefig(output_roc_path, dpi=300)
    plt.close()

    # 2. PR Curves
    plt.figure(figsize=(9, 7))
    sns.set_theme(style="whitegrid")
    for cls in unique_classes:
        if cls < y_proba.shape[1]:
            y_binary = (y_true == cls).astype(int)
            p, r, _ = precision_recall_curve(y_binary, y_proba[:, cls])
            pr_auc_val = auc(r, p)
            plt.plot(r, p, label=f'Class {cls} (AUC = {pr_auc_val:.2f})')
    plt.title('Multi-Class Precision-Recall (PR) Curve', fontsize=12, fontweight='bold')
    plt.xlabel('Recall', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_pr_path), exist_ok=True)
    plt.savefig(output_pr_path, dpi=300)
    plt.close()
