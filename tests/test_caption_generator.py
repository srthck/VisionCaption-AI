import pytest
from utils.caption_generator import clean_caption

def test_clean_caption_normal_case():
    """Test that startseq and endseq are removed and string is capitalized."""
    raw = "startseq a dog is running endseq"
    cleaned = clean_caption(raw)
    assert cleaned == "A dog is running"

def test_clean_caption_empty_case():
    """Test edge cases with empty or partial tokens."""
    assert clean_caption("startseq endseq") == ""
    assert clean_caption("startseq") == ""
    assert clean_caption("endseq") == ""
    
def test_clean_caption_whitespace():
    """Test that extra whitespace is trimmed."""
    raw = " startseq    a cat     endseq "
    cleaned = clean_caption(raw)
    # The current logic just replaces and strips edges. 
    # Let's ensure it handles it reasonably.
    assert cleaned == "A cat"
