from typing import List

from src.utils.schemas import LogicalDocument, ChunkDocument



def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """
    Split text into word-based chunks with overlap.
    """
    if not text or not text.strip():
        return []

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def chunk_logical_document(
    logical_doc: LogicalDocument,
    chunk_size: int = 400,
    overlap: int = 50
) -> List[ChunkDocument]:
    """
    Convert one LogicalDocument into ChunkDocuments.
    """
    chunk_texts = chunk_text(logical_doc.text, chunk_size=chunk_size, overlap=overlap)
    chunks: List[ChunkDocument] = []

    for idx, chunk_text_value in enumerate(chunk_texts, start=1):
        chunk_id = f"{logical_doc.logical_doc_id}_c{idx}"

        chunk = ChunkDocument(
            chunk_id=chunk_id,
            document_id=logical_doc.document_id,
            document_name=logical_doc.document_name,
            page_number=logical_doc.page_start,
            chunk_index=idx,
            text=chunk_text_value,
            metadata={
                "chunk_id": chunk_id,
                "logical_doc_id": logical_doc.logical_doc_id,
                "document_id": logical_doc.document_id,
                "document_name": logical_doc.document_name,
                "doc_type": logical_doc.doc_type,
                "page_start": logical_doc.page_start,
                "page_end": logical_doc.page_end,
                "pages": logical_doc.pages,
                "chunk_index": idx,
            }
        )
        chunks.append(chunk)

    return chunks


def chunk_logical_documents(
    logical_docs: List[LogicalDocument],
    chunk_size: int = 400,
    overlap: int = 50
) -> List[ChunkDocument]:
    """
    Convert a list of LogicalDocuments into ChunkDocuments.
    """
    all_chunks: List[ChunkDocument] = []

    for logical_doc in logical_docs:
        all_chunks.extend(
            chunk_logical_document(
                logical_doc,
                chunk_size=chunk_size,
                overlap=overlap
            )
        )

    return all_chunks