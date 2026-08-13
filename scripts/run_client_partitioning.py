import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.clients.partitioner import NonIIDPartitioner
from src.utils.seed import set_seed

if __name__ == "__main__":
    set_seed(42)
    print("Executing Phase 5: Non-IID Federated Client Simulation...")

    # Load processed data
    data_dir = "data/processed"
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    print(f"Loaded train data: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    # Create 5 Hospital Clients under Dirichlet non-IID skew (alpha=0.5)
    partitioner = NonIIDPartitioner(num_clients=5, dirichlet_alpha=0.5, seed=42)
    clients = partitioner.create_clients(X_train, y_train, X_val, y_val, X_test, y_test)

    # Validate and export distribution CSV & heatmaps
    summary = partitioner.validate_client_partition(clients, results_dir="results")

    print("\nClient Partitioning Completed Successfully!")
    for c_info in summary["clients_summary"]:
        print(f"  {c_info['client_id']}: Train={c_info['train_size']}, Val={c_info['val_size']}, Test={c_info['test_size']}, Unique Classes={c_info['num_unique_classes']}")
