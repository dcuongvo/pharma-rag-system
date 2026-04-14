from typing import List, Tuple
from sentence_transformers import CrossEncoder

from src.utils.schemas import ChunkDocument


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[ChunkDocument, float]],
        top_k: int = 5
    ) -> List[Tuple[ChunkDocument, float]]:
        """
        Rerank retrieved chunks using a cross-encoder.
        Input candidates are (chunk, initial_score), but reranking uses fresh cross-encoder scores.
        """
        if not candidates:
            return []

        pairs = [(query, chunk.text) for chunk, _ in candidates]
        scores = self.model.predict(pairs)

        reranked = [
            (chunk, float(score))
            for (chunk, _), score in zip(candidates, scores)
        ]

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]