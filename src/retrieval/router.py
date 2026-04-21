import json
import re

from src.utils.llm_factory import get_llm
from src.utils.constants import DOC_TYPES, DOC_TYPE_DESCRIPTIONS
from src.utils.llm_utils import safe_llm_call


def _normalize_doc_type_label(raw: str) -> str:
    s = str(raw or "").strip().lower().replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    return s


def _parse_router_response(raw_text: str) -> dict:
    """
    Gemini often wraps JSON in ```json fences or adds a short preamble.
    json.loads on the full string then fails and the router returned unknown, 0.0.
    """
    text = raw_text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


class QueryRouter:
    def __init__(self, provider=None, model=None):
        self.llm = get_llm(provider=provider, model=model)

    def predict(self, query: str):
        doc_type_guide = "\n".join(
            f"- {doc_type}: {description}"
            for doc_type, description in DOC_TYPE_DESCRIPTIONS.items()
        )

        prompt = f"""
        You are a pharmaceutical document router.

        Your task is to predict which document type is MOST likely to contain the answer.

        Document types:
        {doc_type_guide}

        Rules:
        - Choose exactly one label from this list: {DOC_TYPES}
        - Use "unknown" only if the query is ambiguous or does not clearly match a known type
        - Return one JSON object only (no markdown, no code fences, no other text)
        - Confidence must be a number between 0.0 and 1.0

        Return JSON:
        {{"doc_type": "<one valid label>", "confidence": 0.0}}

        Query:
        {query}
        """

        try:
            response = safe_llm_call(self.llm, prompt)
            result = _parse_router_response(response.text)

            doc_type = _normalize_doc_type_label(result.get("doc_type", "unknown"))
            confidence = float(result.get("confidence", 0.0))

            if doc_type not in DOC_TYPES:
                return "unknown", 0.0

            confidence = max(0.0, min(1.0, confidence))
            return doc_type, confidence

        except Exception:
            return "unknown", 0.0