import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_reproducibility_audit import run_independent_reproducibility_audit

def test_reproducibility_audit():
    """Verify independent reproducibility audit executes and verifies all 12 dimensions."""
    report_md = run_independent_reproducibility_audit()
    assert os.path.exists("reports/reproducibility_report.md")
    assert len(report_md) > 1000

    assert "OVERALL STATUS: 100% REPRODUCIBLE (PASS)" in report_md

    required_steps = [
        "Environment Installation", "Dataset Acquisition", "Preprocessing Pipeline",
        "Client Creation", "Centralized Model", "Local Hospital Models",
        "FedAvg Simulation", "Continual Learning", "Zero-Day Detection",
        "Proposed Framework", "Graphs Generation", "Reported Results Verification"
    ]

    for step in required_steps:
        assert step in report_md, f"Missing reproducibility step: {step}"
