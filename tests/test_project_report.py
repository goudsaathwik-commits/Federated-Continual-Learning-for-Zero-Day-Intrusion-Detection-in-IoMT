import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.generate_project_report import compile_academic_project_report

def test_project_report_generation():
    """Verify project report is compiled and contains all 32 required academic sections."""
    report_md = compile_academic_project_report()
    assert os.path.exists("reports/project_report.md")
    assert len(report_md) > 2000

    required_sections = [
        "1. Abstract", "2. Introduction", "3. Background", "4. Motivation",
        "5. Problem Statement", "6. Research Gap", "7. Objectives", "8. Research Questions",
        "9. Literature Review", "10. Dataset", "11. Data Preprocessing", "12. Leakage Prevention",
        "13. IoMT Threat Model", "14. Hospital Simulation", "15. Non-IID Data Partitioning",
        "16. System Architecture", "17. Federated Learning", "18. Continual Learning Engine",
        "19. Zero-Day Attack Detection Architecture", "20. Experimental Setup", "21. Baselines",
        "22. Evaluation Metrics", "23. Empirical Results", "24. Component Ablation Study",
        "25. Discussion", "26. Security Analysis", "27. Privacy Discussion",
        "28. Limitations", "29. Future Work", "30. Conclusion", "31. References", "32. Appendix"
    ]

    for sec in required_sections:
        assert sec in report_md, f"Missing required academic report section: {sec}"
