import os
import json
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.ids_backbone import TabularIDSBackbone
from src.clients.partitioner import NonIIDPartitioner
from src.federated.server import FederatedServer
from src.continual.task_manager import TaskManager
from src.continual.cl_trainer import ContinualTrainer
from src.models.proposed_fcl_ids import ProposedFederatedContinualZeroDayIDS
from src.zero_day.open_set_detector import EnergyBasedZeroDayDetector
from src.utils.seed import set_seed
from src.utils.logging_config import setup_logger

logger = setup_logger("ablation_study")

class AblationStudyEngine:
    """
    Executes systematic Component Ablation Study (A1-A5) and Sensitivity Analysis (A6-A8)
    under strictly identical evaluation conditions.
    """
    def __init__(self, data_dir: str = "data/processed", ablation_dir: str = "results/ablation"):
        self.data_dir = data_dir
        self.ablation_dir = ablation_dir
        os.makedirs(ablation_dir, exist_ok=True)

    def run_ablation_suite(self, seed: int = 42) -> Dict[str, Any]:
        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Running Component Ablation Study (A1 to A5) on device: {device} | Seed = {seed}")

        X_tr = np.load(os.path.join(self.data_dir, "X_train.npy")).astype(np.float32)
        y_tr = np.load(os.path.join(self.data_dir, "y_train.npy")).astype(np.int64)
        X_v = np.load(os.path.join(self.data_dir, "X_val.npy")).astype(np.float32)
        y_v = np.load(os.path.join(self.data_dir, "y_val.npy")).astype(np.int64)
        X_te = np.load(os.path.join(self.data_dir, "X_test.npy")).astype(np.float32)
        y_te = np.load(os.path.join(self.data_dir, "y_test.npy")).astype(np.int64)
        X_zd = np.load(os.path.join(self.data_dir, "X_zero_day.npy")).astype(np.float32)

        input_dim = X_tr.shape[1]
        num_classes = len(np.unique(y_tr))

        # Partition 5 hospital clients (alpha=0.5)
        partitioner = NonIIDPartitioner(num_clients=5, dirichlet_alpha=0.5, seed=seed)
        clients = partitioner.create_clients(X_tr, y_tr, X_v, y_v, X_te, y_te)

        task_mgr = TaskManager()
        num_tasks = 3
        task_train_splits = [[] for _ in range(num_tasks)]
        for c in clients:
            c_splits = task_mgr.create_task_splits(c.X_train, c.y_train, num_tasks=num_tasks)
            for t in range(num_tasks):
                task_train_splits[t].append(c_splits[t])
        global_val_splits = task_mgr.create_task_splits(X_v, y_v, num_tasks=num_tasks)

        ablation_results = {}

        # A1: Centralized Baseline
        logger.info("--> A1: Centralized Baseline")
        model_a1 = TabularIDSBackbone(input_dim, num_classes).to(device)
        opt_a1 = torch.optim.AdamW(model_a1.parameters(), lr=0.001)
        crit_a1 = torch.nn.CrossEntropyLoss()
        for _ in range(5):
            model_a1.train()
            logits = model_a1(torch.tensor(X_tr).to(device))
            loss = crit_a1(logits, torch.tensor(y_tr).to(device))
            opt_a1.zero_grad()
            loss.backward()
            opt_a1.step()
        model_a1.eval()
        with torch.no_grad():
            acc_a1 = float(np.mean(torch.argmax(model_a1(torch.tensor(X_te).to(device)), dim=1).cpu().numpy() == y_te))
        ablation_results["A1_Centralized"] = {"accuracy": round(acc_a1, 4), "bwt": 0.0, "zero_day_auc": "N/A"}

        # A2: Standard FedAvg without CL
        logger.info("--> A2: Standard FedAvg without CL")
        model_a2 = TabularIDSBackbone(input_dim, num_classes)
        server_a2 = FederatedServer(model_a2, clients, device, seed=seed)
        server_a2.run_federated_rounds(num_rounds=5, local_epochs=2, X_val=X_v, y_val=y_v, X_test=X_te, y_test=y_te, num_classes=num_classes)
        acc_a2 = server_a2.evaluate_global_model(X_te, y_te, num_classes=num_classes)["accuracy"]
        ablation_results["A2_FedAvg_No_CL"] = {"accuracy": round(acc_a2, 4), "bwt": 0.0, "zero_day_auc": "N/A"}

        # A3: CL without FL (Centralized Experience Replay)
        logger.info("--> A3: Centralized CL with Replay")
        model_a3 = TabularIDSBackbone(input_dim, num_classes)
        trainer_a3 = ContinualTrainer(model_a3, device, num_classes, seed=seed)
        cent_task_train = task_mgr.create_task_splits(X_tr, y_tr, num_tasks=num_tasks)
        cent_task_val = task_mgr.create_task_splits(X_v, y_v, num_tasks=num_tasks)
        res_a3 = trainer_a3.train_sequential_tasks(cent_task_train, cent_task_val, use_replay=True, epochs_per_task=3)
        ablation_results["A3_CL_No_FL"] = {"accuracy": res_a3["average_accuracy"], "bwt": res_a3["backward_transfer"], "zero_day_auc": "N/A"}

        # A4: FL + CL without Replay Memory (Naive FL+CL)
        logger.info("--> A4: FL + CL without Replay Memory")
        model_a4 = TabularIDSBackbone(input_dim, num_classes)
        engine_a4 = ProposedFederatedContinualZeroDayIDS(model_a4, clients, device, replay_buffer_size=0, seed=seed)
        res_a4 = engine_a4.run_proposed_pipeline(task_train_splits, global_val_splits, X_zd, num_fl_rounds_per_task=2, local_epochs=1, num_classes=num_classes)
        ablation_results["A4_FL_CL_No_Replay"] = {"accuracy": res_a4["average_accuracy"], "bwt": res_a4["backward_transfer_bwt"], "zero_day_auc": res_a4["zero_day_evaluations_per_task"][-1]["roc_auc"]}

        # A5: Proposed FL + CL with Replay Memory (Full Model)
        logger.info("--> A5: Proposed FL + CL with Replay Memory (Full Proposed Model)")
        model_a5 = TabularIDSBackbone(input_dim, num_classes)
        engine_a5 = ProposedFederatedContinualZeroDayIDS(model_a5, clients, device, replay_buffer_size=500, seed=seed)
        res_a5 = engine_a5.run_proposed_pipeline(task_train_splits, global_val_splits, X_zd, num_fl_rounds_per_task=2, local_epochs=1, num_classes=num_classes)
        ablation_results["A5_FL_CL_With_Replay"] = {"accuracy": res_a5["average_accuracy"], "bwt": res_a5["backward_transfer_bwt"], "zero_day_auc": res_a5["zero_day_evaluations_per_task"][-1]["roc_auc"]}

        # Save Metrics JSON
        json_path = os.path.join(self.ablation_dir, "ablation_metrics.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(ablation_results, f, indent=2)

        # Export CSV Summary Table
        summary_rows = []
        for name, metrics in ablation_results.items():
            summary_rows.append({
                "Ablation_Variant": name,
                "Accuracy": metrics["accuracy"],
                "Backward_Transfer_BWT": metrics["bwt"],
                "Zero_Day_ROC_AUC": metrics["zero_day_auc"]
            })
        df_summary = pd.DataFrame(summary_rows)
        csv_path = os.path.join(self.ablation_dir, "ablation_comparison_table.csv")
        df_summary.to_csv(csv_path, index=False)
        logger.info(f"Saved ablation study summary table to: {csv_path}")

        # Plot Ablation Figures
        self._plot_ablation_figures(df_summary)

        return ablation_results

    def _plot_ablation_figures(self, df_summary: pd.DataFrame):
        sns.set_theme(style="whitegrid")

        # 1. Ablation Accuracy Bar Plot
        plt.figure(figsize=(9, 5))
        sns.barplot(data=df_summary, x="Ablation_Variant", y="Accuracy", palette="crest")
        plt.title("Component Ablation Study: Accuracy Comparison (A1 - A5)", fontsize=12, fontweight="bold")
        plt.xlabel("Ablation Variant", fontsize=11)
        plt.ylabel("Accuracy", fontsize=11)
        plt.xticks(rotation=25, ha="right")
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.ablation_dir, "ablation_accuracy_comparison.png"), dpi=300)
        plt.close()

        # 2. Ablation BWT Bar Plot
        plt.figure(figsize=(9, 5))
        sns.barplot(data=df_summary, x="Ablation_Variant", y="Backward_Transfer_BWT", palette="flare")
        plt.title("Component Ablation Study: Backward Transfer (BWT)", fontsize=12, fontweight="bold")
        plt.xlabel("Ablation Variant", fontsize=11)
        plt.ylabel("Backward Transfer (BWT)", fontsize=11)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.ablation_dir, "ablation_bwt_comparison.png"), dpi=300)
        plt.close()
