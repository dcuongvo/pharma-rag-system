from src.services.session_index_service import SessionIndexService
from src.pipelines.retrieval_pipeline import RetrievalPipeline

folder_path = "data/raw_pdfs"
query = "What is the lot number?"

session_service = SessionIndexService()
session_result = session_service.build_session_index(folder_path)

pipeline = RetrievalPipeline(vector_store=session_result["vector_store"])
results = pipeline.query(query, top_k=5)

print(f"\nQuery: {query}", flush=True)

for chunk, score in results:
    print("\n" + "=" * 60, flush=True)
    print(f"Score: {score:.4f}", flush=True)
    print(f"Chunk ID: {chunk.chunk_id}", flush=True)
    print(f"Document: {chunk.document_name}", flush=True)
    print(f"Doc type: {chunk.metadata.get('doc_type')}", flush=True)
    print(f"Page range: {chunk.metadata.get('page_start')} -> {chunk.metadata.get('page_end')}", flush=True)
    print(f"Pages: {chunk.metadata.get('pages')}", flush=True)
    print(f"Preview: {chunk.text[:250]}", flush=True)