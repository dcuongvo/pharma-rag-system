from typing import List

from src.utils.schemas import PageDocument, LogicalDocument
from src.ingestion.continuation_detector import ContinuationDetector
import uuid

class LogicalDocumentBuilder:
    def __init__(self, detector: ContinuationDetector | None = None):
        self.detector = detector or ContinuationDetector()

    def build_logical_documents(self, pages: List[PageDocument]) -> List[LogicalDocument]:
        if not pages:
            return []

        pages = sorted(pages, key=lambda p: (p.document_name, p.page_number))

        logical_documents: List[LogicalDocument] = []
        current_group: List[PageDocument] = [pages[0]]
        logical_index = 1

        for curr_page in pages[1:]:
            prev_page = current_group[-1]

            same_file = prev_page.document_name == curr_page.document_name
            should_continue = same_file and self.detector.should_continue(prev_page, curr_page)

            if should_continue:
                current_group.append(curr_page)
            else:
                logical_documents.append(self._group_to_logical_doc(current_group, logical_index))
                logical_index += 1
                current_group = [curr_page]

        logical_documents.append(self._group_to_logical_doc(current_group, logical_index))
        return logical_documents

    def _group_to_logical_doc(
        self,
        group: List[PageDocument],
        logical_index: int
    ) -> LogicalDocument:
        first_page = group[0]
        last_page = group[-1]

        merged_text = "\n\n".join(page.cleaned_text for page in group if page.cleaned_text)
        logical_doc_id = f"{first_page.document_id}_ldoc_{logical_index}"
        safe_doc_type = (first_page.doc_type or "unknown").replace(" ", "_")
        uid = uuid.uuid4().hex[:8]

        logical_doc_id = (
            f"{uid}_"
            f"{first_page.document_id}_"
            f"{safe_doc_type}_"
            f"p{first_page.page_number}_{last_page.page_number}"
        )
        return LogicalDocument(
            logical_doc_id=logical_doc_id,
            document_id=first_page.document_id,
            document_name=first_page.document_name,
            doc_type=first_page.doc_type or "unknown",
            page_start=first_page.page_number,
            page_end=last_page.page_number,
            pages=[page.page_number for page in group],
            text=merged_text,
            metadata={
                "logical_doc_id": logical_doc_id,
                "document_id": first_page.document_id,
                "document_name": first_page.document_name,
                "doc_type": first_page.doc_type or "unknown",
                "page_start": first_page.page_number,
                "page_end": last_page.page_number,
                "pages": [page.page_number for page in group],
            }
        )