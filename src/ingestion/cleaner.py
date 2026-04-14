import re
import unicodedata
from ftfy import fix_text


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text.

    Steps:
    1. Fix encoding issues
    2. Normalize unicode
    3. Remove null characters
    4. Collapse repeated whitespace
    """

    if not text:
        return ""

    text = fix_text(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()

    return text