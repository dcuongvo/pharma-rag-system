from src.pipelines.indexing_pipeline import IndexingPipeline
from src.pipelines.retrieval_pipeline import RetrievalPipeline


PDF_FOLDER = "data/raw_pdfs"
QUERY = "What is the lot number?"


def main():
    print("\n" + "=" * 60, flush=True)
    print("STAGE 1: INGESTION", flush=True)
    print("=" * 60, flush=True)

    indexing_pipeline = IndexingPipeline(
        chunk_size=300,
        overlap=50,
    )

    index_result = indexing_pipeline.run(PDF_FOLDER)

    print("\n" + "=" * 60, flush=True)
    print("INDEXING STATS", flush=True)
    print("=" * 60, flush=True)

    for key, value in index_result["stats"].items():
        print(f"{key}: {value}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("STAGE 2: RETRIEVAL", flush=True)
    print("=" * 60, flush=True)

    retrieval_pipeline = RetrievalPipeline(
        vector_store=index_result["vector_store"],
        rerank_score_threshold=-999.0,
    )

    output = retrieval_pipeline.query(
        query_text=QUERY,
        top_k=5,
        retrieve_k=10,
    )

    print("\n" + "=" * 60, flush=True)
    print("FULL END TO END TEST", flush=True)
    print("=" * 60, flush=True)

    print(f"\nQuery: {QUERY}", flush=True)
    print(f"Predicted doc type: {output['predicted_doc_type']}", flush=True)
    print(f"Used doc type: {output['used_doc_type']}", flush=True)
    print(f"Router confidence: {output['confidence']:.2f}", flush=True)
    print(f"Answer confidence: {output.get('answer_confidence', 0.0):.4f}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("FINAL ANSWER", flush=True)
    print("=" * 60, flush=True)
    print(output["answer"], flush=True)

    print("\n" + "=" * 60, flush=True)
    print("SOURCES", flush=True)
    print("=" * 60, flush=True)

    for source in output.get("sources", []):
        print("\n" + "-" * 60, flush=True)
        print(f"Source ID: {source.get('source_id')}", flush=True)
        print(f"Chunk ID: {source.get('chunk_id')}", flush=True)
        print(f"Doc type: {source.get('doc_type')}", flush=True)
        print(
            f"Page range: {source.get('page_start')} -> {source.get('page_end')}",
            flush=True,
        )
        print(f"Score: {source.get('score'):.4f}", flush=True)
        print(f"Preview: {source.get('preview')}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("SUPPORTING CHUNKS", flush=True)
    print("=" * 60, flush=True)

    for chunk, score in output["results"]:
        print("\n" + "-" * 60, flush=True)
        print(f"Rerank score: {score:.4f}", flush=True)
        print(f"Chunk ID: {chunk.chunk_id}", flush=True)
        print(f"Document: {chunk.document_name}", flush=True)
        print(f"Doc type: {chunk.metadata.get('doc_type')}", flush=True)
        print(f"Page range: {chunk.page_start} -> {chunk.page_end}", flush=True)
        print(f"Pages: {chunk.metadata.get('pages')}", flush=True)
        print(f"Preview: {chunk.text[:250]}", flush=True)


if __name__ == "__main__":
    main()