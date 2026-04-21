from src.utils.llm_factory import get_llm
from src.utils.llm_utils import safe_llm_call


class AnswerGenerator:
    def __init__(self, provider=None, model=None):
        self.llm = get_llm(provider=provider, model=model)

    def generate(self, query: str, chunks):
        if not chunks:
            return {
                "answer": "No reliable answer found.",
                "sources": [],
                "confidence": 0.0,
            }

        context_parts = []
        sources = []

        for i, (chunk, score) in enumerate(chunks, start=1):
            doc_type = chunk.metadata.get("doc_type", "unknown")
            page_start = chunk.metadata.get("page_start", "?")
            page_end = chunk.metadata.get("page_end", "?")
            chunk_id = chunk.metadata.get("chunk_id", chunk.chunk_id)

            context_parts.append(
                f"[Source {i} | {doc_type} | pages {page_start}-{page_end} | score {score:.4f}]\n{chunk.text}"
            )

            sources.append({
                "source_id": f"Source {i}",
                "chunk_id": chunk_id,
                "doc_type": doc_type,
                "page_start": page_start,
                "page_end": page_end,
                "score": float(score),
                "preview": chunk.text[:120],
            })

        context = "\n\n".join(context_parts)

        prompt = f"""
            You are a pharmaceutical document assistant.

            Answer the question using ONLY the provided context.

            Rules:
            - If there is one clear answer, return that single answer.
            - If the question is broad or ambiguous and multiple relevant answers exist, return only the top 3 to 5 most relevant answers.
            - Do NOT dump every possible value from the context.
            - Prefer the most relevant values from the highest-ranked chunks.
            - If the answer is not clearly supported, say exactly: No reliable answer found.
            - Mention supporting source labels when helpful.
            - Be concise.

            Question:
            {query}

            Context:
            {context}

            Response format:
            Answer: <your answer>
            """

        response = safe_llm_call(self.llm, prompt)
        answer_text = response.text.strip()

        avg_confidence = sum(score for _, score in chunks) / len(chunks)

        return {
            "answer": answer_text,
            "sources": sources,
            "confidence": avg_confidence,
        }