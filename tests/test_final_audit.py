import os
import sys
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_final_audit import execute_final_audit

def test_final_data_and_experiment_audit():
    """Verify final data and experiment audit passes without any FAIL status."""
    audit_findings = execute_final_audit()

    assert audit_findings["overall_status"] == "PASS"
    assert audit_findings["audits"]["data_integrity"]["status"] == "PASS"
    assert audit_findings["audits"]["zero_day_isolation"]["status"] == "PASS"
    assert audit_findings["audits"]["federated_learning"]["status"] == "PASS"
    assert audit_findings["audits"]["continual_learning"]["status"] == "PASS"
    assert audit_findings["audits"]["empirical_results"]["status"] == "PASS"

    assert os.path.exists("results/final_audit.json")
    assert os.path.exists("reports/final_audit.md")
