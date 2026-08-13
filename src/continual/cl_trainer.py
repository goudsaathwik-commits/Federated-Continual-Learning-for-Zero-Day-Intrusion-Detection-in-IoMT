import copy
import logging
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.ids_backbone import TabularIDSBackbone
from src.continual.replay_buffer import ReplayBuffer
from src.evaluation.metrics import evaluate_classification_metrics
from src.utils.seed import set_seed
from src.utils.logging_config import setup_logger

logger = setup_logger("cl_trainer")

class ContinualTrainer:
    """
    Continual Learning Trainer evaluating Catastrophic Forgetting and Backward Transfer (BWT)
    across sequential task phases. Compares Naive Fine-Tuning vs Experience Replay.
    """
    def __init__(self, model: nn.Module, device: torch.device, num_classes: int, seed: int = 42):
        self.initial_model = copy.deepcopy(model)
        self.device = device
        self.num_classes = num_classes
        self.seed = seed

    def train_sequential_tasks(self, task_train_data: List[Tuple[np.ndarray, np.ndarray]],
                               task_val_data: List[Tuple[np.ndarray, np.ndarray]],
                               use_replay: bool = False, replay_buffer_size: int = 500,
                               epochs_per_task: int = 5, batch_size: int = 64, lr: float = 0.001) -> Dict[str, Any]:
        """
        Executes sequential task training and records accuracy matrix R_{i, j}.
        Returns: Results dictionary containing R matrix, Average Accuracy, Forgetting, and BWT.
        """
        set_seed(self.seed)
        model = copy.deepcopy(self.initial_model).to(self.device)
        replay_buffer = ReplayBuffer(buffer_size=replay_buffer_size, seed=self.seed) if use_replay else None

        num_tasks = len(task_train_data)
        # R_matrix[i, j] stores accuracy on Task j after training on Task i
        R_matrix = np.zeros((num_tasks, num_tasks))

        criterion = nn.CrossEntropyLoss()

        for t_idx in range(num_tasks):
            X_task_train, y_task_train = task_train_data[t_idx]
            logger.info(f"--- Continual Task {t_idx + 1}/{num_tasks} Training (Replay={use_replay}) --- | Samples: {len(y_task_train)}")

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            dataset = TensorDataset(torch.tensor(X_task_train, dtype=torch.float32),
                                    torch.tensor(y_task_train, dtype=torch.int64))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for epoch in range(1, epochs_per_task + 1):
                model.train()
                for bx, by in loader:
                    bx, by = bx.to(self.device), by.to(self.device)

                    # If using Replay and buffer is populated, mix in replay samples
                    if use_replay and not replay_buffer.is_empty():
                        rx, ry = replay_buffer.sample_batch(batch_size=16)
                        rx_tensor = torch.tensor(rx, dtype=torch.float32).to(self.device)
                        ry_tensor = torch.tensor(ry, dtype=torch.int64).to(self.device)

                        bx = torch.cat([bx, rx_tensor], dim=0)
                        by = torch.cat([by, ry_tensor], dim=0)

                    optimizer.zero_grad()
                    logits = model(bx)
                    loss = criterion(logits, by)
                    loss.backward()
                    optimizer.step()

            # If using replay, add representative samples from completed task to buffer
            if use_replay:
                replay_buffer.add_samples(X_task_train, y_task_train, samples_per_class=50)

            # Evaluate model on all tasks learned so far (1..t_idx)
            model.eval()
            for eval_idx in range(num_tasks):
                X_eval, y_eval = task_val_data[eval_idx]
                eval_dataset = TensorDataset(torch.tensor(X_eval, dtype=torch.float32),
                                            torch.tensor(y_eval, dtype=torch.int64))
                eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

                all_preds = []
                with torch.no_grad():
                    for bx, _ in eval_loader:
                        bx = bx.to(self.device)
                        logits = model(bx)
                        preds = torch.argmax(logits, dim=1)
                        all_preds.extend(preds.cpu().numpy())

                acc = float(np.mean(np.array(all_preds) == y_eval))
                R_matrix[t_idx, eval_idx] = acc

            logger.info(f"Task {t_idx + 1} complete. Accuracies across Tasks 1-{num_tasks}: {np.round(R_matrix[t_idx], 4)}")

        # Calculate Continual Metrics
        final_accs = R_matrix[-1, :]
        avg_acc = float(np.mean(final_accs))

        # Backward Transfer (BWT): (1 / (T - 1)) * \sum_{j=1}^{T-1} (R_{T, j} - R_{j, j})
        bwt = float(np.mean([R_matrix[-1, j] - R_matrix[j, j] for j in range(num_tasks - 1)]))

        # Forgetting: f_j = \max_{l \in 1..T-1} R_{l, j} - R_{T, j}
        forgetting_per_task = [float(np.max(R_matrix[:num_tasks-1, j]) - R_matrix[-1, j]) for j in range(num_tasks - 1)]
        avg_forgetting = float(np.mean(forgetting_per_task))

        return {
            "strategy": "Experience Replay" if use_replay else "Sequential Fine-Tuning (Naive)",
            "R_matrix": R_matrix.tolist(),
            "average_accuracy": round(avg_acc, 4),
            "backward_transfer": round(bwt, 4),
            "average_forgetting": round(avg_forgetting, 4),
            "final_task_accuracies": [round(a, 4) for a in final_accs]
        }
