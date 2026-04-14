from src.ingestion.pdf_loader import load_pdfs_from_folder
from src.ingestion.ocr import needs_ocr

folder_path = "data/raw_pdfs"
pages = load_pdfs_from_folder(folder_path)

for page in pages[:15]:
    print("\n" + "=" * 60)
    print(f"Document: {page.document_name}")
    print(f"Page: {page.page_number}")
    print(f"Needs OCR: {needs_ocr(page.cleaned_text)}")

# Test OCR extraction on a specific page
# ======================================== 
print ("\n" + "=" * 60)
from src.ingestion.pdf_loader import load_pdf_pages
from src.ingestion.ocr import needs_ocr, extract_text_with_ocr

pdf_path = "data/raw_pdfs/pharmaceutical-sdf-page4-certificate-processing.pdf"
pages = load_pdf_pages(pdf_path)

for page in pages[:3]:
    print("\n" + "=" * 60)
    print(f"Page: {page.page_number}")
    print(f"PyMuPDF preview: {page.cleaned_text[:200]}")
    print(f"Needs OCR: {needs_ocr(page.cleaned_text)}")

    if needs_ocr(page.cleaned_text):
        ocr_text = extract_text_with_ocr(pdf_path, page.page_number)
        print(f"OCR preview: {ocr_text}")