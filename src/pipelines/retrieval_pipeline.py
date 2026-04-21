from src.embedding.embedder_factory import get_embedder
from src.indexing.vector_store import VectorStore
from src.retrieval.router import QueryRouter
from src.retrieval.reranker import Reranker
from src.retrieval.answer_generator import AnswerGenerator


def _store_ready(store: VectorStore | None) -> bool:
    return (
        store is not None
        and store.index is not None
        and bool(store.chunks)
    )


class RetrievalPipeline:
    def __init__(
        self,
        embedder_type=None,
        embedding_model=None,
        llm_provider=None,
        llm_model=None,
        vector_store: VectorStore | None = None,
        upload_vector_store: VectorStore | None = None,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        routing_conf_threshold: float = 0.7,
        rerank_score_threshold: float = 0.15,
    ):
        self.embedder = get_embedder(
            embedder_type=embedder_type,
            model_name=embedding_model,
        )
        self.vector_store = vector_store or VectorStore(self.embedder)
        self.upload_vector_store = upload_vector_store
        self.router = QueryRouter(provider=llm_provider, model=llm_model)
        self.reranker = Reranker(model_name=reranker_model)
        self.answer_generator = AnswerGenerator(provider=llm_provider, model=llm_model)

        self.routing_conf_threshold = routing_conf_threshold
        self.rerank_score_threshold = rerank_score_threshold

    def _filter_by_rerank_threshold(self, reranked_results):
        return [
            (chunk, score)
            for chunk, score in reranked_results
            if score >= self.rerank_score_threshold
        ]

    def load_index(self, save_dir: str):
        self.vector_store.load(save_dir)

    def _try_retrieve_from_store(
        self,
        store: VectorStore,
        query_text: str,
        top_k: int,
        retrieve_k: int,
        predicted_doc_type: str,
        confidence: float,
    ) -> dict | None:
        if not _store_ready(store):
            return None

        # Step 1: routed retrieval
        if confidence > self.routing_conf_threshold:
            candidates = store.search_by_doc_type(
                query=query_text,
                doc_type=predicted_doc_type,
                top_k=retrieve_k,
            )

            if candidates:
                reranked = self.reranker.rerank(
                    query=query_text,
                    candidates=candidates,
                    top_k=top_k,
                )

                filtered = self._filter_by_rerank_threshold(reranked)

                if filtered:
                    answer_result = self.answer_generator.generate(query_text, filtered)
                    return {
                        "predicted_doc_type": predicted_doc_type,
                        "used_doc_type": predicted_doc_type,
                        "confidence": confidence,
                        "answer": answer_result["answer"],
                        "answer_confidence": answer_result["confidence"],
                        "sources": answer_result["sources"],
                        "results": filtered,
                    }

        # Step 2: fallback retrieval
        candidates = store.search(query_text, top_k=retrieve_k)

        reranked = self.reranker.rerank(
            query=query_text,
            candidates=candidates,
            top_k=top_k,
        )

        filtered = self._filter_by_rerank_threshold(reranked)

        if not filtered:
            return None

        answer_result = self.answer_generator.generate(query_text, filtered)

        return {
            "predicted_doc_type": predicted_doc_type,
            "used_doc_type": "all",
            "confidence": confidence,
            "answer": answer_result["answer"],
            "answer_confidence": answer_result["confidence"],
            "sources": answer_result["sources"],
            "results": filtered,
        }

    def query(self, query_text: str, top_k: int = 5, retrieve_k: int = 10):
        predicted_doc_type, confidence = self.router.predict(query_text)

        print(
            f"Predicted: {predicted_doc_type} (confidence={confidence:.2f})",
            flush=True
        )

        if _store_ready(self.upload_vector_store):
            upload_result = self._try_retrieve_from_store(
                self.upload_vector_store,
                query_text,
                top_k,
                retrieve_k,
                predicted_doc_type,
                confidence,
            )
            if upload_result is not None:
                upload_result["retrieval_scope"] = "uploads"
                return upload_result

        if _store_ready(self.vector_store):
            library_result = self._try_retrieve_from_store(
                self.vector_store,
                query_text,
                top_k,
                retrieve_k,
                predicted_doc_type,
                confidence,
            )
            if library_result is not None:
                library_result["retrieval_scope"] = "library"
                return library_result

        return {
            "predicted_doc_type": predicted_doc_type,
            "used_doc_type": "all",
            "confidence": confidence,
            "answer": "No reliable answer found.",
            "answer_confidence": 0.0,
            "sources": [],
            "results": [],
            "retrieval_scope": "none",
        }
