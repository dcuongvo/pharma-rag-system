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

# -------------------------
# FINAL ANSWER
# -------------------------
print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)
print(output["answer"])

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
    print(
        f"Page range: {chunk.metadata.get('page_start')} -> {chunk.metadata.get('page_end')}"
    )
    print(f"Pages: {chunk.metadata.get('pages')}")
    print(f"Preview: {chunk.text[:250]}")