import os
import json
import logging
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix

from src.utils.logging_config import setup_logger

logger = setup_logger("zero_day_detector")

class EnergyBasedZeroDayDetector:
    r"""
    Explicit Open-Set Anomaly Detector for Zero-Day Intrusion Detection using Energy-Based Scoring.
    Formula: E(x; w) = -T * log( \sum_{i=1}^C exp( g_i(x) / T ) )
    In-distribution known traffic yields lower energy scores than unseen zero-day attacks.
    """
    def __init__(self, model: nn.Module, temperature: float = 1.0, threshold_percentile: float = 95.0):
        self.model = model
        self.temperature = temperature
        self.threshold_percentile = threshold_percentile
        self.threshold: float = 0.0
        self.fitted = False

    def compute_energy_scores(self, X: np.ndarray, device: torch.device) -> np.ndarray:
        """
        Computes free energy scores E(x; w) for feature matrix X.
        """
        self.model.eval()
        self.model.to(device)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        energy_scores = []

        batch_size = 128
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                bx = X_tensor[i : i + batch_size]
                logits = self.model(bx)
                
                # E(x; w) = -T * LogSumExp( logits / T )
                energy = -self.temperature * torch.logsumexp(logits / self.temperature, dim=1)
                energy_scores.extend(energy.cpu().numpy())

        return np.array(energy_scores)

    def fit_threshold(self, X_val_known: np.ndarray, device: torch.device):
        """
        Fits energy decision threshold tau on known in-distribution validation data.
        Sets threshold at the 95th percentile of known validation energy scores.
        """
        logger.info("Fitting Energy-Based Open-Set Detector decision threshold on known validation data...")
        val_energies = self.compute_energy_scores(X_val_known, device)
        
        # High energy indicates Out-Of-Distribution (Zero-Day Attack)
        self.threshold = float(np.percentile(val_energies, self.threshold_percentile))
        self.fitted = True
        logger.info(f"Fitted Zero-Day Energy Threshold (tau at {self.threshold_percentile}%ile): {self.threshold:.4f}")

    def predict_open_set(self, X: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts whether samples are known (0) or Zero-Day Unknown Attack (1) using energy threshold tau.
        Returns: (is_zero_day_pred, energy_scores, closed_set_preds)
        """
        if not self.fitted:
            raise RuntimeError("EnergyBasedZeroDayDetector must be fitted on validation data before predicting!")

        self.model.eval()
        self.model.to(device)

        energies = self.compute_energy_scores(X, device)
        is_zero_day = (energies > self.threshold).astype(int)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        closed_set_preds = []
        with torch.no_grad():
            for i in range(0, len(X), 128):
                bx = X_tensor[i : i + 128]
                logits = self.model(bx)
                preds = torch.argmax(logits, dim=1)
                closed_set_preds.extend(preds.cpu().numpy())

        return is_zero_day, energies, np.array(closed_set_preds)

    def evaluate_zero_day_detection(self, X_known_test: np.ndarray, X_zero_day_test: np.ndarray, device: torch.device) -> Dict[str, Any]:
        """
        Evaluates open-set detection performance comparing Known Test (OOD label 0) vs Withheld Zero-Day Test (OOD label 1).
        Calculates Precision, Recall, F1, FPR, FNR, ROC-AUC, and PR-AUC.
        """
        # Combine Known Test and Zero-Day Test
        X_eval = np.vstack([X_known_test, X_zero_day_test])
        y_eval_ood = np.array([0] * len(X_known_test) + [1] * len(X_zero_day_test))

        pred_ood, energies, _ = self.predict_open_set(X_eval, device)

        prec = float(precision_score(y_eval_ood, pred_ood, zero_division=0))
        rec = float(recall_score(y_eval_ood, pred_ood, zero_division=0))
        f1 = float(f1_score(y_eval_ood, pred_ood, zero_division=0))

        cm = confusion_matrix(y_eval_ood, pred_ood)
        tn, fp, fn, tp = cm.ravel()

        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        roc_auc = float(roc_auc_score(y_eval_ood, energies))
        p_curve, r_curve, _ = precision_recall_curve(y_eval_ood, energies)
        pr_auc = float(auc(r_curve, p_curve))

        return {
            "threshold_tau": self.threshold,
            "zero_day_precision": round(prec, 4),
            "zero_day_recall": round(rec, 4),
            "zero_day_f1": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "confusion_matrix": cm.tolist(),
            "known_test_samples": len(X_known_test),
            "zero_day_test_samples": len(X_zero_day_test)
        }
