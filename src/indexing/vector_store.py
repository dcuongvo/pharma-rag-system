from typing import List, Tuple, Dict
import faiss
import numpy as np
import pickle
from pathlib import Path

from src.embedding.base_embedder import BaseEmbedder
from src.utils.schemas import ChunkDocument


class VectorStore:
    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder

        self.index = None  # global index
        self.doc_type_indices: Dict[str, dict] = {}  # per-doc-type indices

        self.chunks: List[ChunkDocument] = []
        self.dimension: int | None = None

    def build_index(self, chunks: List[ChunkDocument]) -> None:
        if not chunks:
            raise ValueError("No chunks provided to build the index.")

        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]

        embeddings = self.embedder.embed_documents(texts).astype(np.float32)

        self.dimension = embeddings.shape[1]

        # Global index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)

        # Build per-doc-type indices
        self.doc_type_indices = {}
        doc_types = set(chunk.metadata.get("doc_type", "unknown") for chunk in chunks)

        for doc_type in doc_types:
            indices = [
                i for i, chunk in enumerate(chunks)
                if chunk.metadata.get("doc_type", "unknown") == doc_type
            ]

            if not indices:
                continue

            type_embeddings = embeddings[indices]

            type_index = faiss.IndexFlatL2(self.dimension)
            type_index.add(type_embeddings)

            self.doc_type_indices[doc_type] = {
                "index": type_index,
                "mapping": indices  # map back to global chunks
            }

        print(f"Built global index + {len(self.doc_type_indices)} doc_type indices")

    def _search_index(self, index, query_embedding, mapping=None, top_k=5):
        distances, indices = index.search(query_embedding, top_k)

        results = []

        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue

            real_idx = mapping[idx] if mapping else idx
            chunk = self.chunks[real_idx]

            score = 1 / (1 + float(dist))
            results.append((chunk, score))

        return results

    def search(self, query: str, top_k: int = 5):
        if self.index is None:
            raise ValueError("Index not built.")

        query_embedding = self.embedder.embed_query(query)
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        return self._search_index(self.index, query_embedding, top_k=top_k)

    def search_by_doc_type(self, query: str, doc_type: str, top_k: int = 5):
        if doc_type not in self.doc_type_indices:
            return []

        data = self.doc_type_indices[doc_type]

        query_embedding = self.embedder.embed_query(query)
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        return self._search_index(
            data["index"],
            query_embedding,
            mapping=data["mapping"],
            top_k=top_k,
        )

    def save(self, save_dir: str):
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        if self.index is None:
            raise ValueError("No index available to save.")
        faiss.write_index(self.index, str(path / "index.faiss"))

        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, save_dir: str):
        path = Path(save_dir)

        self.index = faiss.read_index(str(path / "index.faiss"))

        with open(path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        self.dimension = self.index.d
        self._rebuild_doc_type_indices()

    def _rebuild_doc_type_indices(self):
        if self.index is None or not self.chunks:
            self.doc_type_indices = {}
            return

        self.doc_type_indices = {}

        for doc_type in set(chunk.metadata.get("doc_type", "unknown") for chunk in self.chunks):
            indices = [
                i for i, chunk in enumerate(self.chunks)
                if chunk.metadata.get("doc_type", "unknown") == doc_type
            ]

            if not indices:
                continue

            vectors = np.vstack([self.index.reconstruct(i) for i in indices]).astype(np.float32)

            type_index = faiss.IndexFlatL2(self.dimension)
            type_index.add(vectors)

            self.doc_type_indices[doc_type] = {
                "index": type_index,
                "mapping": indices,
            }