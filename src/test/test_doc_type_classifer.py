from src.ingestion.pdf_loader import load_pdfs_from_folder
from src.ingestion.doc_type_classifier import DocTypeClassifier

folder_path = "data/raw_pdfs"

print("Loading PDFs...")
pages = load_pdfs_from_folder(folder_path)

print("\nClassifying document types...")
classifier = DocTypeClassifier()
pages = classifier.attach_doc_type_to_pages(pages)

print(f"\nTotal pages: {len(pages)}")

for page in pages[:10]:
    print("\n" + "=" * 60)
    print(f"Document: {page.document_name}")
    print(f"Page: {page.page_number}")
    print(f"Extraction method: {page.extraction_method}")
    print(f"Doc type: {page.doc_type}")
    print(f"Metadata doc_type: {page.metadata.get('doc_type')}")
    print(f"Preview: {page.cleaned_text[:250]}")