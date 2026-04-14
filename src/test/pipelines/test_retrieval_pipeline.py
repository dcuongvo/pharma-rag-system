from src.pipelines.retrieval_pipeline import RetrievalPipeline

save_dir = "storage/pharma_index"
query = "What is the lot number?"

pipeline = RetrievalPipeline()
pipeline.load_index(save_dir)

output = pipeline.query(query_text=query, top_k=5, retrieve_k=10)

print(f"\nQuery: {query}", flush=True)
print(f"Used doc type: {output['doc_type']}", flush=True)
print(f"Router confidence: {output['confidence']:.2f}", flush=True)

print("\n" + "=" * 60, flush=True)
print("FINAL ANSWER", flush=True)
print(output["answer"], flush=True)

print("\n" + "=" * 60, flush=True)
print("SUPPORTING CHUNKS", flush=True)

for chunk, score in output["results"]:
    print("\n" + "=" * 60, flush=True)
    print(f"Rerank score: {score:.4f}", flush=True)
    print(f"Chunk ID: {chunk.chunk_id}", flush=True)
    print(f"Document: {chunk.document_name}", flush=True)
    print(f"Doc type: {chunk.metadata.get('doc_type')}", flush=True)
    print(f"Page range: {chunk.metadata.get('page_start')} -> {chunk.metadata.get('page_end')}", flush=True)
    print(f"Pages: {chunk.metadata.get('pages')}", flush=True)
    print(f"Preview: {chunk.text[:250]}", flush=True)