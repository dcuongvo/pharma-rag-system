from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass
class PageDocument:
    document_id: str
    document_name: str
    page_number: int
    raw_text: str
    cleaned_text: str = ""
    extraction_method: str = "pymupdf"
    doc_type: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class LogicalDocument:
    logical_doc_id: str
    document_id: str
    document_name: str
    doc_type: str
    page_start: int
    page_end: int
    pages: List[int]
    text: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class ChunkDocument:
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    chunk_index: int
    text: str
    metadata: Dict = field(default_factory=dict)