import copy
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.clients.hospital_client import HospitalClient
from src.federated.fedavg import aggregate_fedavg
from src.continual.replay_buffer import ReplayBuffer
from src.zero_day.open_set_detector import EnergyBasedZeroDayDetector
from src.evaluation.metrics import evaluate_classification_metrics
from src.utils.seed import set_seed
from src.utils.logging_config import setup_logger

logger = setup_logger("proposed_fcl_ids")

class ProposedFederatedContinualZeroDayIDS:
    """
    Proposed Unified Research Engine:
    Integrates Federated Learning (FedAvg) + Continual Learning (Experience Replay)
    + Non-IID Hospital Clients + Energy-Based Zero-Day Open-Set Anomaly Detection.
    """
    def __init__(self, global_model: nn.Module, clients: List[HospitalClient],
                 device: torch.device, replay_buffer_size: int = 500, seed: int = 42):
        self.global_model = global_model
        self.clients = clients
        self.device = device
        self.replay_buffer_size = replay_buffer_size
        self.seed = seed
        set_seed(self.seed)

        # Initialize local replay buffers per hospital client
        self.client_buffers = [
            ReplayBuffer(buffer_size=replay_buffer_size, seed=seed + i)
            for i in range(len(clients))
        ]

    def train_client_local_task(self, client: HospitalClient, client_buffer: ReplayBuffer,
                                global_weights: Dict[str, torch.Tensor],
                                X_task: np.ndarray, y_task: np.ndarray,
                                local_epochs: int = 3, batch_size: int = 64, lr: float = 0.001) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Executes local Continual SGD training for a client on new task data,
        mixing in historical replay samples from the client's local memory buffer.
        """
        local_model = copy.deepcopy(self.global_model)
        local_model.load_state_dict(global_weights)
        local_model.to(self.device)
        local_model.train()

        dataset = TensorDataset(torch.tensor(X_task, dtype=torch.float32),
                                torch.tensor(y_task, dtype=torch.int64))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(local_model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(local_epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)

                # Mix in historical replay samples if buffer is populated
                if not client_buffer.is_empty():
                    rx, ry = client_buffer.sample_batch(batch_size=16)
                    rx_t = torch.tensor(rx, dtype=torch.float32).to(self.device)
                    ry_t = torch.tensor(ry, dtype=torch.int64).to(self.device)

                    bx = torch.cat([bx, rx_t], dim=0)
                    by = torch.cat([by, ry_t], dim=0)

                optimizer.zero_grad()
                logits = local_model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        # Update local client replay memory with new task samples
        client_buffer.add_samples(X_task, y_task, samples_per_class=40)

        return local_model.cpu().state_dict(), len(y_task)

    def run_proposed_pipeline(self, task_train_splits: List[List[Tuple[np.ndarray, np.ndarray]]],
                              task_val_splits: List[Tuple[np.ndarray, np.ndarray]],
                              X_zero_day_test: np.ndarray,
                              num_fl_rounds_per_task: int = 3,
                              local_epochs: int = 2,
                              batch_size: int = 64,
                              lr: float = 0.001,
                              num_classes: int = 12) -> Dict[str, Any]:
        """
        Executes full proposed Federated Continual Zero-Day pipeline across sequential tasks.
        Returns: Task performance matrices, Zero-Day AUROC, and BWT metrics.
        """
        num_tasks = len(task_train_splits)
        R_matrix = np.zeros((num_tasks, num_tasks))
        zero_day_evals = []

        logger.info(f"Running Proposed FCL Zero-Day Pipeline across {num_tasks} Continual Tasks | FL Rounds per Task: {num_fl_rounds_per_task}")

        for t_idx in range(num_tasks):
            logger.info(f"\n=================== PROPOSED TASK PHASE {t_idx + 1}/{num_tasks} ===================")

            # FL Rounds for Current Task Phase
            for fl_round in range(1, num_fl_rounds_per_task + 1):
                global_weights = copy.deepcopy(self.global_model.state_dict())
                client_weights = []
                sample_counts = []

                for c_idx, client in enumerate(self.clients):
                    c_X_task, c_y_task = task_train_splits[t_idx][c_idx]
                    if len(c_y_task) == 0:
                        continue

                    updated_weights, n_k = self.train_client_local_task(
                        client=client,
                        client_buffer=self.client_buffers[c_idx],
                        global_weights=global_weights,
                        X_task=c_X_task,
                        y_task=c_y_task,
                        local_epochs=local_epochs,
                        batch_size=batch_size,
                        lr=lr
                    )
                    client_weights.append(updated_weights)
                    sample_counts.append(n_k)

                # Aggregation using FedAvg
                new_global_weights = aggregate_fedavg(client_weights, sample_counts)
                self.global_model.load_state_dict(new_global_weights)

            # Evaluate Global Model on all tasks learned so far (1..t_idx)
            self.global_model.eval()
            self.global_model.to(self.device)

            for eval_idx in range(num_tasks):
                X_v, y_v = task_val_splits[eval_idx]
                v_dataset = TensorDataset(torch.tensor(X_v, dtype=torch.float32), torch.tensor(y_v, dtype=torch.int64))
                v_loader = DataLoader(v_dataset, batch_size=batch_size, shuffle=False)

                all_preds = []
                with torch.no_grad():
                    for bx, _ in v_loader:
                        bx = bx.to(self.device)
                        logits = self.global_model(bx)
                        preds = torch.argmax(logits, dim=1)
                        all_preds.extend(preds.cpu().numpy())

                task_acc = float(np.mean(np.array(all_preds) == y_v))
                R_matrix[t_idx, eval_idx] = task_acc

            # Evaluate Energy-Based Zero-Day Detection at end of task
            X_val_known, _ = task_val_splits[t_idx]
            detector = EnergyBasedZeroDayDetector(model=self.global_model, threshold_percentile=95.0)
            detector.fit_threshold(X_val_known, device=self.device)
            zd_metrics = detector.evaluate_zero_day_detection(X_known_test=X_val_known, X_zero_day_test=X_zero_day_test, device=self.device)
            zero_day_evals.append(zd_metrics)

            logger.info(f"Task {t_idx + 1} FL-CL Complete. Accuracies 1-{num_tasks}: {np.round(R_matrix[t_idx], 4)} | Zero-Day ROC-AUC: {zd_metrics['roc_auc']:.4f}")

        # Compute Continual Metrics
        final_accs = R_matrix[-1, :]
        avg_acc = float(np.mean(final_accs))
        bwt = float(np.mean([R_matrix[-1, j] - R_matrix[j, j] for j in range(num_tasks - 1)]))

        return {
            "proposed_R_matrix": R_matrix.tolist(),
            "average_accuracy": round(avg_acc, 4),
            "backward_transfer_bwt": round(bwt, 4),
            "final_task_accuracies": [round(a, 4) for a in final_accs],
            "zero_day_evaluations_per_task": zero_day_evals
        }
