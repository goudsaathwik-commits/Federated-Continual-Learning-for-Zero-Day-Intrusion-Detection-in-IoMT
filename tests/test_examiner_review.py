import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_examiner_review import compile_phd_examiner_review

def test_examiner_review_generation():
    """Verify PhD examiner review report is compiled, evaluates 22 dimensions, challenges 6 claims, and classifies as RESEARCH READY."""
    review_md = compile_phd_examiner_review()
    assert os.path.exists("reports/final_examiner_review.md")
    assert len(review_md) > 2000

    assert "CLASSIFICATION: RESEARCH READY" in review_md

    # Check 6 challenged claims
    claims = [
        "The system detects zero-day attacks.",
        "The system is privacy-preserving.",
        "The system is suitable for IoMT.",
        "Federated Learning improves performance.",
        "Continual Learning prevents catastrophic forgetting.",
        "The proposed method is better than existing methods."
    ]

    for claim in claims:
        assert claim[:25] in review_md, f"Missing challenged claim: {claim}"

    # Check 22 dimensions present in text
    dimensions = [
        "Dataset Selection", "IoMT Domain Justification", "Problem Formulation",
        "Data Preprocessing", "Leakage Prevention", "Non-IID Hospital Simulation",
        "Federated Learning Mechanics", "FedAvg Aggregation Correctness", "Continual Learning Engine",
        "Catastrophic Forgetting Mitigation", "Zero-Day Attack Detection", "Open-Set Recognition",
        "Baselines", "Component Ablation", "Metrics Selection", "Statistical Validity",
        "Reproducibility", "Security Assumptions", "Privacy Claims", "Research Contribution",
        "Limitations Analysis", "Practical Relevance"
    ]

    for dim in dimensions:
        assert dim in review_md, f"Missing reviewer evaluation dimension: {dim}"
