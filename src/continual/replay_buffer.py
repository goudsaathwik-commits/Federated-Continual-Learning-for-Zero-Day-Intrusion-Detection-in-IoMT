import numpy as np
from typing import Tuple, List, Optional

class ReplayBuffer:
    """
    Experience Replay Memory Buffer for Continual Learning.
    Stores a bounded reservoir of representative samples from previously completed tasks.
    STRICT SECURITY RULE: Never stores zero-day attack samples or test set data.
    """
    def __init__(self, buffer_size: int = 500, seed: int = 42, withheld_class_ids: Optional[List[int]] = None):
        self.buffer_size = buffer_size
        self.seed = seed
        self.withheld_class_ids = withheld_class_ids if withheld_class_ids is not None else [-1]

        self.X_buffer: Optional[np.ndarray] = None
        self.y_buffer: Optional[np.ndarray] = None

    def is_empty(self) -> bool:
        return self.X_buffer is None or len(self.X_buffer) == 0

    def __len__(self) -> int:
        return 0 if self.is_empty() else len(self.X_buffer)

    def add_samples(self, X_task: np.ndarray, y_task: np.ndarray, samples_per_class: int = 50):
        """
        Populates memory buffer with balanced random samples per class from completed task training split.
        Rejects any samples belonging to zero-day withheld classes.
        """
        np.random.seed(self.seed)

        # Filter out zero-day samples
        valid_mask = ~np.isin(y_task, self.withheld_class_ids)
        X_valid = X_task[valid_mask]
        y_valid = y_task[valid_mask]

        selected_X = []
        selected_y = []

        unique_classes = np.unique(y_valid)
        for cls in unique_classes:
            cls_mask = (y_valid == cls)
            cls_indices = np.where(cls_mask)[0]
            n_select = min(samples_per_class, len(cls_indices))

            chosen_idx = np.random.choice(cls_indices, size=n_select, replace=False)
            selected_X.append(X_valid[chosen_idx])
            selected_y.append(y_valid[chosen_idx])

        if not selected_X:
            return

        new_X = np.vstack(selected_X)
        new_y = np.concatenate(selected_y)

        if self.is_empty():
            self.X_buffer = new_X
            self.y_buffer = new_y
        else:
            self.X_buffer = np.vstack([self.X_buffer, new_X])
            self.y_buffer = np.concatenate([self.y_buffer, new_y])

        # Enforce maximum capacity constraint via uniform random sampling
        if len(self.X_buffer) > self.buffer_size:
            keep_indices = np.random.choice(len(self.X_buffer), size=self.buffer_size, replace=False)
            self.X_buffer = self.X_buffer[keep_indices]
            self.y_buffer = self.y_buffer[keep_indices]

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples a random mini-batch of historical samples from the replay buffer.
        """
        if self.is_empty():
            raise RuntimeError("Cannot sample from an empty ReplayBuffer!")

        actual_batch = min(batch_size, len(self.X_buffer))
        sample_idx = np.random.choice(len(self.X_buffer), size=actual_batch, replace=False)

        return self.X_buffer[sample_idx], self.y_buffer[sample_idx]
