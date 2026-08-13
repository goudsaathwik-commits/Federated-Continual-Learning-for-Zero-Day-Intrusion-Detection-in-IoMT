import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.generate_presentation import build_presentation_markdown

def test_presentation_slides_generation():
    """Verify 20-slide academic presentation is generated with speaker notes and figure embeds."""
    slides_md = build_presentation_markdown()
    assert os.path.exists("presentation/project_presentation.md")
    assert len(slides_md) > 2000

    for slide_num in range(1, 21):
        slide_heading = f"Slide {slide_num}:"
        assert slide_heading in slides_md, f"Missing slide {slide_num} heading in presentation"

    # Verify speaker notes exist
    assert "Speaker Notes" in slides_md
    assert slides_md.count("Speaker Notes") >= 20
