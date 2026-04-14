# Test for PDF loading and page extraction
from src.ingestion.pdf_loader import load_pdf_pages

pdf_path = "data/raw_pdfs/sample-sdf-document.pdf"
pages = load_pdf_pages(pdf_path)

print(f"Loaded {len(pages)} pages")

for page in pages[:3]:
    print("\n" + "=" * 60)
    print(f"Document: {page.document_name}")
    print(f"Page: {page.page_number}")
    print(f"Raw length: {len(page.raw_text)}")
    print(f"Cleaned length: {len(page.cleaned_text)}")
    print(f"Preview: {page.cleaned_text[:300]}")


#Test for loading multiple PDFs from a folder
print("\n" + "=" * 100)
print("=" * 50)
print("Testing loading multiple PDFs from folder")
print
from src.ingestion.pdf_loader import load_pdfs_from_folder

folder_path = "data/raw_pdfs"

pages = load_pdfs_from_folder(folder_path)

print(f"\nTotal pages loaded: {len(pages)}")

# show first few pages
for page in pages[:5]:
    print("\n" + "=" * 60)
    print(f"Document: {page.document_name}")
    print(f"Page: {page.page_number}")
    print(f"Raw length: {len(page.raw_text)}")
    print(f"Cleaned length: {len(page.cleaned_text)}")
    print(f"Preview: {page.cleaned_text[:200]}")