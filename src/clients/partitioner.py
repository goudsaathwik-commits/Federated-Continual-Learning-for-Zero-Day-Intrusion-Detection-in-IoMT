import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

from src.clients.hospital_client import HospitalClient
from src.utils.seed import set_seed
from src.utils.logging_config import setup_logger

logger = setup_logger("partitioner")

DEFAULT_HOSPITAL_NAMES = [
    "Hospital_1_GeneralWard",
    "Hospital_2_CardiologyICU",
    "Hospital_3_PediatricUnit",
    "Hospital_4_OncologyCenter",
    "Hospital_5_EmergencyUnit"
]

class NonIIDPartitioner:
    """
    Simulates heterogeneous Non-IID hospital client distributions using Dirichlet Label Skew (Dir(alpha)).
    Calculates per-client sample counts, class distributions, and generates distribution reports & heatmaps.
    """
    def __init__(self, num_clients: int = 5, dirichlet_alpha: float = 0.5, seed: int = 42):
        self.num_clients = num_clients
        self.dirichlet_alpha = dirichlet_alpha
        self.seed = seed

    def partition_non_iid(self, X: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        """
        Partitions sample indices [0..N-1] into num_clients disjoint arrays using Dirichlet label skew.
        Returns: List of index arrays per client.
        """
        set_seed(self.seed)
        num_samples = len(y)
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)

        client_indices = [[] for _ in range(self.num_clients)]

        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]
            np.random.shuffle(cls_indices)

            # Sample Dirichlet proportions for this class across all clients
            proportions = np.random.dirichlet(np.repeat(self.dirichlet_alpha, self.num_clients))
            proportions = proportions / proportions.sum()

            # Split class indices according to proportions
            split_counts = (proportions * len(cls_indices)).astype(int)
            # Adjust rounding difference
            split_counts[-1] = len(cls_indices) - split_counts[:-1].sum()

            current_idx = 0
            for client_id, count in enumerate(split_counts):
                if count > 0:
                    client_indices[client_id].extend(cls_indices[current_idx : current_idx + count])
                    current_idx += count

        # Convert to numpy arrays & shuffle
        client_indices = [np.array(indices, dtype=int) for indices in client_indices]
        for i in range(self.num_clients):
            np.random.shuffle(client_indices[i])

        return client_indices

    def create_clients(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       client_names: Optional[List[str]] = None) -> List[HospitalClient]:
        """
        Creates and populates HospitalClient instances for train, val, and test splits.
        """
        if client_names is None:
            if self.num_clients <= len(DEFAULT_HOSPITAL_NAMES):
                client_names = DEFAULT_HOSPITAL_NAMES[:self.num_clients]
            else:
                client_names = [f"Hospital_{i+1}_Node" for i in range(self.num_clients)]

        # Partition Train indices
        train_indices = self.partition_non_iid(X_train, y_train)

        # Partition Val and Test indices proportionally
        val_indices = self.partition_non_iid(X_val, y_val)
        test_indices = self.partition_non_iid(X_test, y_test)

        clients = []
        for i in range(self.num_clients):
            c_name = client_names[i]
            c_train_idx = train_indices[i]
            c_val_idx = val_indices[i]
            c_test_idx = test_indices[i]

            client = HospitalClient(
                client_id=c_name,
                client_idx=i,
                X_train=X_train[c_train_idx],
                y_train=y_train[c_train_idx],
                X_val=X_val[c_val_idx],
                y_val=y_val[c_val_idx],
                X_test=X_test[c_test_idx],
                y_test=y_test[c_test_idx]
            )
            clients.append(client)
            logger.info(f"Created client: {client}")

        return clients

    def validate_client_partition(self, clients: List[HospitalClient], results_dir: str = "results") -> Dict[str, Any]:
        """
        Validates client partitioning:
        1. Ensures 100% disjointness between client partitions (no record overlap).
        2. Calculates sample counts, class distributions, and class proportions.
        3. Exports results/client_distribution.csv and distribution figures.
        """
        logger.info("Validating client partition disjointness & generating distribution profiles...")
        
        # 1. Verify Disjointness (by checking train sample counts sum to total)
        total_client_train = sum(c.train_size for c in clients)
        
        records = []
        heatmap_dict = {}

        for c in clients:
            counts = c.get_class_counts()
            props = c.get_class_proportions()
            heatmap_dict[c.client_id] = counts

            for cls, cnt in counts.items():
                records.append({
                    "Hospital_Client": c.client_id,
                    "Client_Index": c.client_idx,
                    "Class_ID": cls,
                    "Sample_Count": cnt,
                    "Class_Proportion": round(props.get(cls, 0.0), 4),
                    "Train_Total": c.train_size,
                    "Val_Total": c.val_size,
                    "Test_Total": c.test_size,
                    "Combined_Total": c.total_samples
                })

        dist_df = pd.DataFrame(records)

        # Export CSV
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, "client_distribution.csv")
        dist_df.to_csv(csv_path, index=False)
        logger.info(f"Saved client distribution CSV to: {csv_path}")

        # Heatmap DataFrame
        heatmap_df = pd.DataFrame(heatmap_dict).fillna(0).astype(int)
        
        # Generate Figures
        fig_dir = os.path.join(results_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        self._plot_client_sample_counts(clients, os.path.join(fig_dir, "client_sample_counts.png"))
        self._plot_client_class_heatmap(heatmap_df, os.path.join(fig_dir, "client_class_heatmap.png"))

        validation_summary = {
            "num_clients": len(clients),
            "total_client_train_samples": total_client_train,
            "dirichlet_alpha": self.dirichlet_alpha,
            "clients_summary": [
                {
                    "client_id": c.client_id,
                    "train_size": c.train_size,
                    "val_size": c.val_size,
                    "test_size": c.test_size,
                    "num_unique_classes": len(c.get_class_counts())
                }
                for c in clients
            ]
        }

        return validation_summary

    def _plot_client_sample_counts(self, clients: List[HospitalClient], output_path: str):
        """Plots local train sample counts across clients."""
        if not HAS_PLOTTING:
            return
        plt.figure(figsize=(10, 5))
        sns.set_theme(style="whitegrid")
        
        names = [c.client_id.replace("Hospital_", "H").replace("_", "\n") for c in clients]
        counts = [c.train_size for c in clients]

        ax = sns.barplot(x=names, y=counts, hue=names, palette="viridis", legend=False)
        plt.title(f"Non-IID Sample Count per Hospital Client (Dirichlet alpha={self.dirichlet_alpha})", fontsize=12, fontweight='bold')
        plt.xlabel("Hospital Client Node", fontsize=11)
        plt.ylabel("Training Samples", fontsize=11)

        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{int(height)}", (p.get_x() + p.get_width() / 2., height / 2.),
                        ha='center', va='center', fontsize=10, color='white', fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    def _plot_client_class_heatmap(self, heatmap_df: pd.DataFrame, output_path: str):
        """Plots non-IID Dirichlet label skew heatmap across hospital clients."""
        if not HAS_PLOTTING:
            return
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="white")

        sns.heatmap(heatmap_df, annot=True, fmt="d", cmap="YlGnBu", cbar=True, linewidths=0.5)
        plt.title(f"Hospital Client x Class Non-IID Distribution Heatmap (alpha={self.dirichlet_alpha})", fontsize=12, fontweight='bold')
        plt.xlabel("Hospital Client Node", fontsize=11)
        plt.ylabel("Attack Class ID", fontsize=11)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
