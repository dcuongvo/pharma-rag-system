from src.embedding.embedder_factory import get_embedder
from src.indexing.vector_store import VectorStore
from src.retrieval.router import QueryRouter
from src.retrieval.reranker import Reranker
from src.retrieval.answer_generator import AnswerGenerator

class RetrievalPipeline:
    def __init__(
        self,
        embedder_type=None,
        embedding_model=None,
        llm_provider=None,
        llm_model=None,
        vector_store: VectorStore | None = None,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.embedder = get_embedder(
            embedder_type=embedder_type,
            model_name=embedding_model,
        )
        self.vector_store = vector_store or VectorStore(self.embedder)
        self.router = QueryRouter(provider=llm_provider, model=llm_model)
        self.reranker = Reranker(model_name=reranker_model)
        self.answer_generator = AnswerGenerator(provider=llm_provider, model=llm_model)
        
    def load_index(self, save_dir: str):
        self.vector_store.load(save_dir)

    def query(self, query_text: str, top_k: int = 5, retrieve_k: int = 10):
        predicted_doc_type, confidence = self.router.predict(query_text)

        print(
            f"Predicted: {predicted_doc_type} (confidence={confidence:.2f})",
            flush=True
        )

        # Step 1: routed retrieval
        if confidence > 0.7:
            candidates = self.vector_store.search_by_doc_type(
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
                answer = self.answer_generator.generate(query_text, reranked)

                return {
                    "predicted_doc_type": predicted_doc_type,
                    "used_doc_type": predicted_doc_type,
                    "confidence": confidence,
                    "answer": answer,
                    "results": reranked,
                }

        # Step 2: fallback retrieval
        candidates = self.vector_store.search(query_text, top_k=retrieve_k)

        reranked = self.reranker.rerank(
            query=query_text,
            candidates=candidates,
            top_k=top_k
        )
        answer = self.answer_generator.generate(query_text, reranked)

        return {
            "predicted_doc_type": predicted_doc_type,
            "used_doc_type": "all",
            "confidence": confidence,
            "answer": answer,
            "results": reranked,
        }