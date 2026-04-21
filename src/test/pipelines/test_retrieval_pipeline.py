from src.pipelines.retrieval_pipeline import RetrievalPipeline

# Path to your saved FAISS index
save_dir = "storage/pharma_index"

# Test query
query = "What is the lot number?"

# Initialize pipeline
pipeline = RetrievalPipeline()
pipeline.load_index(save_dir)

# Run full pipeline
output = pipeline.query(
    query_text=query,
    top_k=5,        # final results after reranking
    retrieve_k=10   # candidates before reranking
)

# =========================
# PRINT RESULTS
# =========================

print("\n" + "=" * 60)
print("FULL RAG PIPELINE TEST")
print("=" * 60)

print(f"\nQuery: {query}")
print(f"Predicted doc type: {output['predicted_doc_type']}")
print(f"Used doc type: {output['used_doc_type']}")
print(f"Router confidence: {output['confidence']:.2f}")
print(f"Answer confidence: {output.get('answer_confidence', 0.0):.4f}")

# -------------------------
# FINAL ANSWER
# -------------------------
print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)
print(output["answer"])

# -------------------------
# SOURCES
# -------------------------
print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)

for source in output.get("sources", []):
    print("\n" + "-" * 60)
    print(f"Source ID: {source.get('source_id')}")
    print(f"Chunk ID: {source.get('chunk_id')}")
    print(f"Doc type: {source.get('doc_type')}")
    print(f"Page range: {source.get('page_start')} -> {source.get('page_end')}")
    print(f"Score: {source.get('score'):.4f}")
    print(f"Preview: {source.get('preview')}")

# -------------------------
# SUPPORTING CHUNKS
# -------------------------
print("\n" + "=" * 60)
print("SUPPORTING CHUNKS")
print("=" * 60)

for chunk, score in output["results"]:
    print("\n" + "-" * 60)
    print(f"Rerank score: {score:.4f}")
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Document: {chunk.document_name}")
    print(f"Doc type: {chunk.metadata.get('doc_type')}")
    print(f"Chunk page_start: {chunk.page_start}")
    print(f"Chunk page_end: {chunk.page_end}")
    print(
        f"Metadata page range: {chunk.metadata.get('page_start')} -> {chunk.metadata.get('page_end')}"
    )
    print(f"Pages: {chunk.metadata.get('pages')}")
    print(f"Preview: {chunk.text[:250]}")