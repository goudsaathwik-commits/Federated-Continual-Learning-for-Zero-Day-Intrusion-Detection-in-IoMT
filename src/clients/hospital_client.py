import numpy as np
from typing import Dict, Any

class HospitalClient:
    """
    Simulated Hospital Client Node in a Federated Learning environment.
    Encapsulates local data partitions (X_train, y_train, X_val, y_val, X_test, y_test)
    and client metadata.
    """
    def __init__(self, client_id: str, client_idx: int,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray):
        self.client_id = client_id
        self.client_idx = client_idx
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

    @property
    def train_size(self) -> int:
        return len(self.y_train)

    @property
    def val_size(self) -> int:
        return len(self.y_val)

    @property
    def test_size(self) -> int:
        return len(self.y_test)

    @property
    def total_samples(self) -> int:
        return self.train_size + self.val_size + self.test_size

    def get_class_counts(self) -> Dict[int, int]:
        """Calculates training set class counts."""
        unique, counts = np.unique(self.y_train, return_counts=True)
        return {int(cls): int(cnt) for cls, cnt in zip(unique, counts)}

    def get_class_proportions(self) -> Dict[int, float]:
        """Calculates training set class proportions."""
        counts = self.get_class_counts()
        total = sum(counts.values())
        if total == 0:
            return {}
        return {cls: cnt / total for cls, cnt in counts.items()}

    def __repr__(self) -> str:
        return f"<HospitalClient {self.client_id} (idx={self.client_idx}): Train={self.train_size}, Val={self.val_size}, Test={self.test_size}>"
