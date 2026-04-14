from src.pipelines.retrieval_pipeline import RetrievalPipeline

save_dir = "storage/pharma_index"
query = "What is the lot number?"

pipeline = RetrievalPipeline()
pipeline.load_index(save_dir)

output = pipeline.query(query, top_k=5)

print(f"\nQuery: {query}")
print(f"Doc type: {output['doc_type']} (confidence={output['confidence']:.2f})")

for chunk, score in output["results"]:
    print("\n" + "=" * 60)
    print(f"Rerank Score: {score:.4f}")
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Doc type: {chunk.metadata.get('doc_type')}")
    print(f"Preview: {chunk.text[:250]}")