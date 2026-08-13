import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc
)

def evaluate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                     y_proba: Optional[np.ndarray] = None,
                                     num_classes: Optional[int] = None) -> Dict[str, Any]:
    """
    Computes comprehensive intrusion detection classification metrics.
    Includes Accuracy, Precision, Recall, F1 (macro & weighted), Balanced Acc,
    Specificity, FPR, FNR, ROC-AUC, and PR-AUC.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Confusion matrix per-class breakdown to calculate Specificity, FPR, and FNR
    cm = confusion_matrix(y_true, y_pred)
    
    specificities = []
    fprs = []
    fnrs = []

    for i in range(len(cm)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fp + fn)

        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        specificities.append(spec)
        fprs.append(fpr)
        fnrs.append(fnr)

    mean_specificity = float(np.mean(specificities))
    mean_fpr = float(np.mean(fprs))
    mean_fnr = float(np.mean(fnrs))

    metrics = {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "balanced_accuracy": bal_acc,
        "specificity_macro": mean_specificity,
        "false_positive_rate": mean_fpr,
        "false_negative_rate": mean_fnr,
        "confusion_matrix": cm.tolist()
    }

    # Calculate ROC-AUC & PR-AUC if probabilities are provided
    if y_proba is not None:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] > 1:
                # One-vs-Rest ROC-AUC
                roc_auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
                metrics["roc_auc_macro"] = roc_auc
            else:
                metrics["roc_auc_macro"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        except Exception:
            metrics["roc_auc_macro"] = 0.0

        # PR-AUC Macro Average
        pr_aucs = []
        unique_labels = np.unique(y_true)
        for cls in unique_labels:
            if y_proba.ndim == 2 and cls < y_proba.shape[1]:
                y_binary = (y_true == cls).astype(int)
                p, r, _ = precision_recall_curve(y_binary, y_proba[:, cls])
                pr_auc_val = auc(r, p)
                if not np.isnan(pr_auc_val):
                    pr_aucs.append(pr_auc_val)
        if pr_aucs:
            metrics["pr_auc_macro"] = float(np.mean(pr_aucs))
        else:
            metrics["pr_auc_macro"] = 0.0

    return metrics
