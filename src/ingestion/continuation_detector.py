import re
from src.utils.schemas import PageDocument
from src.utils.llm_factory import get_llm
from src.utils.llm_utils import safe_indexing_llm_call

class ContinuationDetector:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        use_llm_fallback: bool = True,
    ):
        self.use_llm_fallback = use_llm_fallback
        self.llm = get_llm(provider=provider, model=model) if use_llm_fallback else None

    def extract_doc_number(self, text: str) -> str | None:
        if not text:
            return None

        patterns = [
            r"document\s+(?:no|number)\.?:?\s*([A-Z0-9\-_\/]+)",
            r"certificate\s+(?:no|number)\.?:?\s*([A-Z0-9\-_\/]+)",
            r"process run id\s*:?\s*([A-Z0-9\-_\/]+)",
            r"doc(?:ument)?\s*no\.?:?\s*([A-Z0-9\-_\/]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip().upper()

        return None

    def has_continuation_signal(self, text: str) -> bool:
        if not text:
            return False

        t = text.lower()

        signals = [
            "continued",
            "page 2 of",
            "page 3 of",
            "page 4 of",
            "page 5 of",
            "continuation",
        ]

        return any(signal in t for signal in signals)

    def continuation_score(self, prev_page: PageDocument, curr_page: PageDocument) -> int:
        score = 0

        if prev_page.document_name == curr_page.document_name:
            score += 1

        if curr_page.page_number == prev_page.page_number + 1:
            score += 2

        if prev_page.doc_type == curr_page.doc_type and prev_page.doc_type != "unknown":
            score += 2

        prev_doc_num = self.extract_doc_number(prev_page.cleaned_text)
        curr_doc_num = self.extract_doc_number(curr_page.cleaned_text)
        if prev_doc_num and curr_doc_num and prev_doc_num == curr_doc_num:
            score += 3

        if self.has_continuation_signal(curr_page.cleaned_text):
            score += 3

        if prev_page.doc_type != curr_page.doc_type:
            score -= 3

        return score

    def llm_same_document(self, prev_page: PageDocument, curr_page: PageDocument) -> bool:
        prev_sample = prev_page.cleaned_text[-700:] if prev_page.cleaned_text else ""
        curr_sample = curr_page.cleaned_text[:700] if curr_page.cleaned_text else ""

        prompt = f"""
            You are deciding whether two consecutive pharmaceutical document pages belong to the SAME logical document.

            Previous page doc type: {prev_page.doc_type or "unknown"}
            Current page doc type: {curr_page.doc_type or "unknown"}

            Use clues such as:
            - same title or heading
            - same document/certificate number
            - same product or process run id
            - explicit continuation phrases
            - whether the second page clearly starts a new document

            End of previous page:
            {prev_sample}

            Start of current page:
            {curr_sample}

            Answer ONLY one word:
            yes
            or
            no
        """

        response = safe_indexing_llm_call(self.llm, prompt)
        answer = response.text.strip().lower()
        return answer.startswith("yes")

    def should_continue(self, prev_page: PageDocument, curr_page: PageDocument) -> bool:
        score = self.continuation_score(prev_page, curr_page)

        if score >= 5:
            return True

        if score <= 0:
            return False

        if self.use_llm_fallback and self.llm is not None:
            return self.llm_same_document(prev_page, curr_page)

        return False