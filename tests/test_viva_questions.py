import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.generate_viva_questions import compile_viva_qa_document

def test_viva_questions_generation():
    """Verify viva QA document is compiled, contains 100+ questions, and covers all 25 categories."""
    qa_md = compile_viva_qa_document()
    assert os.path.exists("viva/viva_questions.md")
    assert len(qa_md) > 3000

    # Count questions (Q1:, Q2:, ...)
    q_count = qa_md.count("### Q")
    assert q_count >= 100, f"Expected at least 100 viva questions, found {q_count}"

    # Verify all 25 categories A through Y
    categories = [
        "Category A:", "Category B:", "Category C:", "Category D:", "Category E:",
        "Category F:", "Category G:", "Category H:", "Category I:", "Category J:",
        "Category K:", "Category L:", "Category M:", "Category N:", "Category O:",
        "Category P:", "Category Q:", "Category R:", "Category S:", "Category T:",
        "Category U:", "Category V:", "Category W:", "Category X:", "Category Y:"
    ]

    for cat in categories:
        assert cat in qa_md, f"Missing viva category: {cat}"
