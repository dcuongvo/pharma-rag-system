from src.ingestion.pdf_loader import load_pdfs_from_folder
from src.ingestion.doc_type_classifier import DocTypeClassifier
from src.ingestion.logical_document_builder import LogicalDocumentBuilder
from src.ingestion.chunker import chunk_logical_documents

from src.embedding.embedder_factory import get_embedder
from src.indexing.vector_store import VectorStore

folder_path = "data/raw_pdfs"

print("Loading PDFs...", flush=True)
pages = load_pdfs_from_folder(folder_path)

print("Classifying doc types...", flush=True)
classifier = DocTypeClassifier()
pages = classifier.attach_doc_type_to_pages(pages)

print("Building logical documents...", flush=True)
builder = LogicalDocumentBuilder()
logical_docs = builder.build_logical_documents(pages)

print("Chunking logical documents...", flush=True)
chunks = chunk_logical_documents(logical_docs, chunk_size=300, overlap=50)

print("Loading embedder...", flush=True)
embedder = get_embedder()

print("Building vector store...", flush=True)
vector_store = VectorStore(embedder=embedder)
vector_store.build_index(chunks)

print(f"Indexed {len(chunks)} chunks", flush=True)

query = "What is the lot number?"
print(f"\nQuery: {query}", flush=True)

results = vector_store.search(query, top_k=5)

for chunk, score in results:
    print("\n" + "=" * 60, flush=True)
    print(f"Score: {score:.4f}", flush=True)
    print(f"Chunk ID: {chunk.chunk_id}", flush=True)
    print(f"Document: {chunk.document_name}", flush=True)
    print(f"Doc type: {chunk.metadata.get('doc_type')}", flush=True)
    print(f"Page range: {chunk.metadata.get('page_start')} -> {chunk.metadata.get('page_end')}", flush=True)
    print(f"Pages: {chunk.metadata.get('pages')}", flush=True)
    print(f"Preview: {chunk.text[:250]}", flush=True)