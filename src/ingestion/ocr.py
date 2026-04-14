import re
from pathlib import Path
import fitz
import pytesseract
from PIL import Image

from src.ingestion.cleaner import clean_text


def needs_ocr(
    text: str,
    min_chars: int = 50,
    min_words: int = 10,
    weird_char_threshold: float = 0.30
) -> bool:
    """
    Decide whether a page likely needs OCR.

    OCR is needed if:
    1. Text is empty or too short
    2. Too few words
    3. Too many unusual characters
    """
    if not text:
        return True

    stripped = text.strip()

    # Too short
    if len(stripped) < min_chars:
        return True

    # Too few words
    words = stripped.split()
    if len(words) < min_words:
        return True

    # Too many weird characters
    weird_chars = re.findall(r"[^a-zA-Z0-9\s.,:/()%-]", stripped)
    weird_ratio = len(weird_chars) / max(len(stripped), 1)

    return weird_ratio > weird_char_threshold


def extract_text_with_ocr(pdf_path: str, page_number: int, dpi: int = 200) -> str:
    """
    Render one PDF page as an image and extract text with Tesseract OCR.

    Args:
        pdf_path: Path to the PDF file
        page_number: 1-based page number
        dpi: Render resolution for OCR

    Returns:
        Cleaned OCR text
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)

    try:
        page = doc[page_number - 1]
        pix = page.get_pixmap(dpi=dpi)

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        ocr_text = pytesseract.image_to_string(image)
        return clean_text(ocr_text)

    finally:
        doc.close()