import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.continual.task_manager import TaskManager
from src.continual.replay_buffer import ReplayBuffer

def test_task_manager_splitting():
    """Verify TaskManager splits data into discrete sequential tasks."""
    X = np.random.randn(300, 10)
    y = np.array([0]*100 + [1]*50 + [2]*50 + [3]*50 + [4]*50)

    manager = TaskManager()
    tasks = manager.create_task_splits(X, y, num_tasks=3)

    assert len(tasks) == 3
    for X_t, y_t in tasks:
        assert len(X_t) > 0
        assert 0 in np.unique(y_t) # Normal class 0 present in all tasks

def test_replay_buffer_zero_day_rejection():
    """Verify ReplayBuffer strictly rejects zero-day withheld class IDs (-1)."""
    buffer = ReplayBuffer(buffer_size=100, seed=42, withheld_class_ids=[-1, 99])

    X_sample = np.random.randn(50, 10)
    y_sample = np.array([0]*30 + [-1]*10 + [99]*10) # 20 valid zero-day samples

    buffer.add_samples(X_sample, y_sample, samples_per_class=50)

    assert len(buffer) == 30 # Only class 0 samples added
    assert -1 not in buffer.y_buffer
    assert 99 not in buffer.y_buffer
