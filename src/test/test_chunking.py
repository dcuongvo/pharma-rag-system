from src.ingestion.pdf_loader import load_pdf_pages, load_pdfs_from_folder
from src.ingestion.doc_type_classifier import DocTypeClassifier
from src.ingestion.chunker import chunk_pages

file_path = "data/raw_pdfs/pharma-blob-sample.pdf"

print("Loading PDFs...")
pages = load_pdf_pages(file_path)

print("\nClassifying doc types...")
classifier = DocTypeClassifier()
pages = classifier.attach_doc_type_to_pages(pages)

print("\nChunking pages...")
chunks = chunk_pages(pages, chunk_size=120, overlap=20)

print(f"\nTotal pages: {len(pages)}")
print(f"Total chunks: {len(chunks)}")

for chunk in chunks[:10]:
    print("\n" + "=" * 60)
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Document: {chunk.document_name}")
    print(f"Page: {chunk.page_number}")
    print(f"Chunk index: {chunk.chunk_index}")
    print(f"Doc type: {chunk.metadata.get('doc_type')}")
    print(f"Extraction method: {chunk.metadata.get('extraction_method')}")
    print(f"Text preview: {chunk.text[:250]}")