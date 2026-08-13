import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any

class TaskManager:
    """
    Manages sequential task streams for Continual Learning on Edge-IIoTset.
    Partitioning known attack classes into discrete sequential task phases:
      Task 1 (T1): Normal + DoS/DDoS (DoS_UDP, DoS_ICMP, DDoS_UDP, DDoS_ICMP)
      Task 2 (T2): Normal + MitM & Injection (ARP_spoofing, DNS_spoofing, SQL_injection, XSS)
      Task 3 (T3): Normal + Scanning & Reconnaissance (Vulnerability_scanner, Port_Scanning)
    """
    def __init__(self, target_label_mapping: Dict[str, int] = None):
        # Default Task Class Mappings (Class IDs corresponding to preprocessed label encoder)
        self.num_tasks = 3

    def create_task_splits(self, X: np.ndarray, y: np.ndarray, num_tasks: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Splits dataset into num_tasks sequential task partitions based on class ID groupings.
        Each task retains Normal class (0) + a subset of attack categories.
        """
        unique_classes = np.unique(y)
        # Separate normal class (0) from attack classes
        attack_classes = [c for c in unique_classes if c != 0]

        # Partition attack classes evenly across num_tasks
        split_attacks = np.array_split(attack_classes, num_tasks)

        task_data = []
        for t_idx, att_group in enumerate(split_attacks):
            task_classes = set([0] + list(att_group))
            mask = np.isin(y, list(task_classes))

            X_task = X[mask]
            y_task = y[mask]
            task_data.append((X_task, y_task))

        return task_data
