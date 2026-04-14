from typing import List
from src.utils.schemas import PageDocument
from src.utils.llm_factory import get_llm
from src.utils.constants import DOC_TYPES
from src.utils.llm_utils import safe_indexing_llm_call



class DocTypeClassifier:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.llm = get_llm(provider=provider, model=model)

    def classify_doc_type(self, text: str) -> str:
        if not text or not text.strip():
            return "unknown"

        prompt = f"""
        You are classifying a pharmaceutical supporting document page.

        Choose exactly one label from this list:
        {", ".join(DOC_TYPES)}

        Rules:
        - Return only the label
        - Do not explain
        - If uncertain, return unknown

        Page text:
        {text[:4000]}
        """

        response = safe_indexing_llm_call(self.llm, prompt)
        label = response.text.strip().lower()

        if label not in DOC_TYPES:
            return "unknown"

        return label

    def attach_doc_type(self, page: PageDocument) -> PageDocument:
        doc_type = self.classify_doc_type(page.cleaned_text)
        page.doc_type = doc_type
        page.metadata["doc_type"] = doc_type
        return page

    def attach_doc_type_to_pages(self, pages: List[PageDocument]) -> List[PageDocument]:
        return [self.attach_doc_type(page) for page in pages]