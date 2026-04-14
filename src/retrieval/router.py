import json
from src.utils.llm_factory import get_llm
from src.utils.constants import DOC_TYPES
from src.utils.llm_utils import safe_llm_call
class QueryRouter:
    def __init__(self, provider=None, model=None):
        self.llm = get_llm(provider=provider, model=model)

    def predict(self, query: str):
        prompt = f"""
Predict the most relevant document type for this query.

Return JSON:
{{"doc_type": "...", "confidence": 0.0-1.0}}

Query:
{query}
"""

        try:
            response = safe_llm_call(self.llm, prompt)
            result = json.loads(response.text.strip())

            doc_type = result.get("doc_type", "unknown")
            confidence = result.get("confidence", 0.0)

            if doc_type not in DOC_TYPES:
                return "unknown", 0.0

            return doc_type, confidence

        except:
            return "unknown", 0.0