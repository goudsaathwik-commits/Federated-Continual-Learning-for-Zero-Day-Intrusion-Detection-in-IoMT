import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.generate_research_paper import compile_ieee_research_paper

def test_ieee_research_paper_generation():
    """Verify IEEE research paper is compiled and contains all required sections."""
    paper_md = compile_ieee_research_paper()
    assert os.path.exists("reports/research_paper.md")
    assert len(paper_md) > 1500

    required_sections = [
        "Abstract", "Keywords", "I. Introduction", "II. Related Work",
        "III. Problem Formulation", "IV. Dataset and Threat Model",
        "V. Proposed Architecture", "VI. Federated Learning",
        "VII. Continual Learning", "VIII. Zero-Day Detection",
        "IX. Experimental Setup", "X. Results", "XI. Ablation Study",
        "XII. Discussion", "XIII. Limitations", "XIV. Conclusion", "References"
    ]

    for sec in required_sections:
        assert sec in paper_md, f"Missing required IEEE section: {sec}"
