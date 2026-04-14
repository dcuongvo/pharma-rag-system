from src.ingestion.pdf_loader import load_pdfs_from_folder
from src.ingestion.doc_type_classifier import DocTypeClassifier
from src.ingestion.logical_document_builder import LogicalDocumentBuilder

folder_path = "data/raw_pdfs"

print("Loading PDFs...", flush=True)
pages = load_pdfs_from_folder(folder_path)

print("Classifying doc types...", flush=True)
classifier = DocTypeClassifier()
pages = classifier.attach_doc_type_to_pages(pages)

print("Building logical documents...", flush=True)
builder = LogicalDocumentBuilder()
logical_docs = builder.build_logical_documents(pages)

print(f"\nTotal pages: {len(pages)}", flush=True)
print(f"Total logical documents: {len(logical_docs)}", flush=True)

for doc in logical_docs[:10]:
    print("\n" + "=" * 60, flush=True)
    print(f"Logical Doc ID: {doc.logical_doc_id}", flush=True)
    print(f"Document: {doc.document_name}", flush=True)
    print(f"Doc type: {doc.doc_type}", flush=True)
    print(f"Pages: {doc.pages}", flush=True)
    print(f"Page range: {doc.page_start} -> {doc.page_end}", flush=True)
    print(f"Preview: {doc.text[:250]}", flush=True)