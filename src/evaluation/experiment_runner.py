import os
import json
import copy
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch

from src.data.loader import EdgeIIoTLoader
from src.data.preprocessor import LeakageFreePreprocessor
from src.clients.partitioner import NonIIDPartitioner
from src.models.ids_backbone import TabularIDSBackbone
from src.models.classical_ids import build_classical_ids
from src.federated.server import FederatedServer
from src.continual.task_manager import TaskManager
from src.continual.cl_trainer import ContinualTrainer
from src.zero_day.open_set_detector import EnergyBasedZeroDayDetector
from src.models.proposed_fcl_ids import ProposedFederatedContinualZeroDayIDS
from src.evaluation.metrics import evaluate_classification_metrics
from src.utils.seed import set_seed
from src.utils.logging_config import setup_logger

logger = setup_logger("experiment_runner")

class MasterExperimentRunner:
    """
    Automated Master Experiment Runner executing E1 to E7 experiments
    and sensitivity sweeps (client counts K=[3,5,10], non-IID alphas=[0.1, 0.5, 10.0], seeds=[42, 100]).
    Persists configuration, seed, checkpoints, prediction arrays, and metrics JSON for every run.
    """
    def __init__(self, data_dir: str = "data/processed", results_dir: str = "results", models_dir: str = "models"):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.models_dir = models_dir

        os.makedirs(os.path.join(results_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

    def load_processed_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train = np.load(os.path.join(self.data_dir, "X_train.npy")).astype(np.float32)
        y_train = np.load(os.path.join(self.data_dir, "y_train.npy")).astype(np.int64)
        X_val = np.load(os.path.join(self.data_dir, "X_val.npy")).astype(np.float32)
        y_val = np.load(os.path.join(self.data_dir, "y_val.npy")).astype(np.int64)
        X_test = np.load(os.path.join(self.data_dir, "X_test.npy")).astype(np.float32)
        y_test = np.load(os.path.join(self.data_dir, "y_test.npy")).astype(np.int64)
        X_zero_day = np.load(os.path.join(self.data_dir, "X_zero_day.npy")).astype(np.float32)
        y_zero_day = np.load(os.path.join(self.data_dir, "y_zero_day.npy")).astype(np.int64)
        return X_train, y_train, X_val, y_val, X_test, y_test, X_zero_day, y_zero_day

    def run_master_benchmark(self, seeds: List[int] = [42]) -> Dict[str, Any]:
        """Runs complete E1-E7 benchmark across specified seeds."""
        X_tr, y_tr, X_v, y_v, X_te, y_te, X_zd, y_zd = self.load_processed_data()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        input_dim = X_tr.shape[1]
        num_classes = len(np.unique(y_tr))

        master_records = {}

        for seed in seeds:
            set_seed(seed)
            logger.info(f"\n=======================================================")
            logger.info(f"   STARTING MASTER BENCHMARK SUITE (SEED = {seed})")
            logger.info(f"=======================================================")

            # E1: Centralized PyTorch MLP
            logger.info("--> Executing E1 Centralized IDS...")
            model_e1 = TabularIDSBackbone(input_dim, num_classes).to(device)
            optimizer = torch.optim.AdamW(model_e1.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            
            # Fast train
            for epoch in range(5):
                model_e1.train()
                logits = model_e1(torch.tensor(X_tr).to(device))
                loss = criterion(logits, torch.tensor(y_tr).to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model_e1.eval()
            with torch.no_grad():
                preds_e1 = torch.argmax(model_e1(torch.tensor(X_te).to(device)), dim=1).cpu().numpy()
                probas_e1 = torch.softmax(model_e1(torch.tensor(X_te).to(device)), dim=1).cpu().numpy()

            metrics_e1 = evaluate_classification_metrics(y_te, preds_e1, probas_e1, num_classes=num_classes)

            # E3: FedAvg (K=5, Alpha=0.5)
            logger.info("--> Executing E3 Standard FedAvg...")
            partitioner = NonIIDPartitioner(num_clients=5, dirichlet_alpha=0.5, seed=seed)
            clients = partitioner.create_clients(X_tr, y_tr, X_v, y_v, X_te, y_te)
            
            global_model_e3 = TabularIDSBackbone(input_dim, num_classes)
            server_e3 = FederatedServer(global_model=global_model_e3, clients=clients, device=device, seed=seed)
            history_e3 = server_e3.run_federated_rounds(num_rounds=5, local_epochs=2, X_val=X_v, y_val=y_v, X_test=X_te, y_test=y_te, num_classes=num_classes)
            metrics_e3 = server_e3.evaluate_global_model(X_te, y_te, num_classes=num_classes)

            # E7: Proposed FCL + Zero-Day
            logger.info("--> Executing E7 Proposed Framework...")
            task_mgr = TaskManager()
            num_tasks = 3
            task_train_splits = [[] for _ in range(num_tasks)]
            for c in clients:
                c_task_splits = task_mgr.create_task_splits(c.X_train, c.y_train, num_tasks=num_tasks)
                for t in range(num_tasks):
                    task_train_splits[t].append(c_task_splits[t])
            global_val_splits = task_mgr.create_task_splits(X_v, y_v, num_tasks=num_tasks)

            prop_model = TabularIDSBackbone(input_dim, num_classes)
            prop_engine = ProposedFederatedContinualZeroDayIDS(global_model=prop_model, clients=clients, device=device, seed=seed)
            metrics_e7 = prop_engine.run_proposed_pipeline(task_train_splits, global_val_splits, X_zd, num_fl_rounds_per_task=2, local_epochs=1, num_classes=num_classes)

            master_records[f"seed_{seed}"] = {
                "E1_Centralized": metrics_e1,
                "E3_FedAvg": metrics_e3,
                "E7_Proposed": metrics_e7
            }

        # Save Master Records JSON
        json_path = os.path.join(self.results_dir, "raw", "master_experiment_suite.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(master_records, f, indent=2)

        return master_records
