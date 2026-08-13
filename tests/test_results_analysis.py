import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analyze_results import generate_results_analysis_report

def test_results_analysis_report_generation():
    """Verify results analysis report is generated and contains all 15 required research sections."""
    report_md = generate_results_analysis_report()
    assert os.path.exists("reports/results_analysis.md")
    assert len(report_md) > 1000

    required_sections = [
        "1. Centralized Baseline Performance",
        "2. Local Hospital Baseline Performance",
        "3. Standard FedAvg Performance",
        "4. Effect of Non-IID Data Distribution",
        "5. Continual Learning Performance",
        "6. Catastrophic Forgetting Analysis",
        "7. Effect of Local Experience Replay Memory",
        "8. Proposed Federated Continual Learning",
        "9. Zero-Day Open-Set Detection Performance",
        "10. False Positive Rate",
        "11. False Negative Rate",
        "12. Communication Overhead",
        "13. Computational Training Cost",
        "14. Best-Performing Method Summary",
        "15. Limitations & Future Work"
    ]

    for sec in required_sections:
        assert sec in report_md, f"Missing required research section: {sec}"
