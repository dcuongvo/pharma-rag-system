from pathlib import Path
from typing import List
import fitz

from src.utils.schemas import PageDocument
from src.ingestion.cleaner import clean_text
from src.ingestion.ocr import needs_ocr, extract_text_with_ocr

def load_pdf_pages(pdf_path: str) -> List[PageDocument]:
    """
    Load a single PDF and return PageDocument list.
    Apply OCR only when extracted text quality is poor.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        raw_text = page.get_text("text")
        cleaned = clean_text(raw_text)

        extraction_method = "pymupdf"

        # Apply OCR only if extracted text looks poor
        if needs_ocr(cleaned):
            print(f"OCR applied on {pdf_path.name} page {i + 1}")
            ocr_text = extract_text_with_ocr(str(pdf_path), i + 1)

            if ocr_text:
                raw_text = ocr_text
                cleaned = ocr_text
                extraction_method = "ocr"

        page_doc = PageDocument(
            document_id=pdf_path.stem,
            document_name=pdf_path.name,
            page_number=i + 1,
            raw_text=raw_text,
            cleaned_text=cleaned,
            extraction_method=extraction_method,
            metadata={
                "source_file": str(pdf_path),
                "page_number": i + 1,
                "document_name": pdf_path.name,
                "ocr_applied": extraction_method == "ocr",
            }
        )

        pages.append(page_doc)

    doc.close()
    return pages


def load_pdfs_from_folder(folder_path: str) -> List[PageDocument]:
    """
    Load all PDFs from a folder.
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in folder.")

    all_pages = []

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")

        pages = load_pdf_pages(str(pdf_file))
        all_pages.extend(pages)

    return all_pages