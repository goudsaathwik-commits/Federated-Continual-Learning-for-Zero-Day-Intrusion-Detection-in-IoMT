import copy
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.clients.hospital_client import HospitalClient
from src.federated.fedavg import aggregate_fedavg
from src.evaluation.metrics import evaluate_classification_metrics
from src.utils.seed import set_seed
from src.utils.logging_config import setup_logger

logger = setup_logger("federated_server")

class FederatedServer:
    """
    Federated Learning Aggregation Server managing FedAvg rounds,
    client broadcast/upload, local SGD execution, and global evaluation.
    """
    def __init__(self, global_model: nn.Module, clients: List[HospitalClient],
                 device: torch.device, seed: int = 42):
        self.global_model = global_model
        self.clients = clients
        self.device = device
        self.seed = seed
        set_seed(self.seed)

        # Calculate single model payload size in Megabytes (32-bit floats)
        num_params = sum(p.numel() for p in global_model.parameters())
        self.model_size_mb = (num_params * 4) / (1024 * 1024)

    def train_local_client(self, client: HospitalClient, global_weights: Dict[str, torch.Tensor],
                          local_epochs: int, batch_size: int, lr: float) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Executes local SGD training on a single hospital client node for E epochs.
        """
        local_model = copy.deepcopy(self.global_model)
        local_model.load_state_dict(global_weights)
        local_model.to(self.device)
        local_model.train()

        dataset = TensorDataset(torch.tensor(client.X_train, dtype=torch.float32),
                                torch.tensor(client.y_train, dtype=torch.int64))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(local_model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(local_epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = local_model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        # Return updated state dict & sample count n_k
        return local_model.cpu().state_dict(), client.train_size

    def evaluate_global_model(self, X_eval: np.ndarray, y_eval: np.ndarray,
                               num_classes: int, batch_size: int = 64) -> Dict[str, Any]:
        """Evaluates current global model on a given dataset split."""
        self.global_model.eval()
        self.global_model.to(self.device)

        dataset = TensorDataset(torch.tensor(X_eval, dtype=torch.float32),
                                torch.tensor(y_eval, dtype=torch.int64))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_preds = []
        all_probas = []

        with torch.no_grad():
            for bx, _ in loader:
                bx = bx.to(self.device)
                logits = self.global_model(bx)
                probas = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probas.extend(probas.cpu().numpy())

        metrics = evaluate_classification_metrics(
            y_true=y_eval,
            y_pred=np.array(all_preds),
            y_proba=np.array(all_probas),
            num_classes=num_classes
        )
        return metrics

    def run_federated_rounds(self, num_rounds: int = 10, fraction_fit: float = 1.0,
                             local_epochs: int = 3, batch_size: int = 64, lr: float = 0.001,
                             X_val: np.ndarray = None, y_val: np.ndarray = None,
                             X_test: np.ndarray = None, y_test: np.ndarray = None,
                             num_classes: int = 12) -> List[Dict[str, Any]]:
        """
        Runs complete FedAvg execution across num_rounds.
        Records round-by-round loss, accuracy, F1, and communication cost.
        """
        logger.info(f"Starting FedAvg simulation across {num_rounds} rounds | Clients: {len(self.clients)} | Local Epochs: {local_epochs}")

        history = []
        cumulative_comm_mb = 0.0

        for r in range(1, num_rounds + 1):
            set_seed(self.seed + r)
            
            # 1. Select participating clients
            num_selected = max(1, int(len(self.clients) * fraction_fit))
            selected_indices = np.random.choice(len(self.clients), size=num_selected, replace=False)
            selected_clients = [self.clients[idx] for idx in selected_indices]

            # Broadcast current global weights
            current_global_weights = copy.deepcopy(self.global_model.state_dict())

            client_weights = []
            sample_counts = []

            # 2. Local Training on Selected Clients
            for client in selected_clients:
                updated_weights, n_k = self.train_local_client(
                    client=client,
                    global_weights=current_global_weights,
                    local_epochs=local_epochs,
                    batch_size=batch_size,
                    lr=lr
                )
                client_weights.append(updated_weights)
                sample_counts.append(n_k)

            # Communication cost (Download + Upload per selected client)
            round_comm_mb = self.model_size_mb * 2 * num_selected
            cumulative_comm_mb += round_comm_mb

            # 3. Aggregate Weights via FedAvg
            new_global_weights = aggregate_fedavg(client_weights, sample_counts)
            self.global_model.load_state_dict(new_global_weights)

            # 4. Evaluate Global Model on Validation & Test Sets
            val_metrics = self.evaluate_global_model(X_val, y_val, num_classes=num_classes, batch_size=batch_size) if X_val is not None else {}
            test_metrics = self.evaluate_global_model(X_test, y_test, num_classes=num_classes, batch_size=batch_size) if X_test is not None else {}

            round_record = {
                "round": r,
                "participating_clients": [c.client_id for c in selected_clients],
                "num_participating": num_selected,
                "round_comm_mb": round(round_comm_mb, 4),
                "cumulative_comm_mb": round(cumulative_comm_mb, 4),
                "val_accuracy": val_metrics.get("accuracy", 0.0),
                "val_f1_macro": val_metrics.get("f1_macro", 0.0),
                "test_accuracy": test_metrics.get("accuracy", 0.0),
                "test_f1_macro": test_metrics.get("f1_macro", 0.0),
                "test_precision_macro": test_metrics.get("precision_macro", 0.0),
                "test_recall_macro": test_metrics.get("recall_macro", 0.0)
            }
            history.append(round_record)

            logger.info(f"Round [{r:02d}/{num_rounds:02d}] Val Acc: {round_record['val_accuracy']:.4f} | Test Acc: {round_record['test_accuracy']:.4f} | Test F1 Macro: {round_record['test_f1_macro']:.4f} | Cum Comm: {cumulative_comm_mb:.2f} MB")

        return history
