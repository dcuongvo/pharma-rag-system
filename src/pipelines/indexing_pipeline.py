from src.ingestion.pdf_loader import load_pdfs_from_folder
from src.ingestion.doc_type_classifier import DocTypeClassifier
from src.ingestion.logical_document_builder import LogicalDocumentBuilder
from src.ingestion.chunker import chunk_logical_documents

from src.embedding.embedder_factory import get_embedder
from src.indexing.vector_store import VectorStore


class IndexingPipeline:
    def __init__(
        self,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        embedder_type: str | None = None,
        embedding_model: str | None = None,
        chunk_size: int = 400,
        overlap: int = 50,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.classifier = DocTypeClassifier(provider=llm_provider, model=llm_model)
        self.logical_doc_builder = LogicalDocumentBuilder()
        self.embedder = get_embedder(embedder_type=embedder_type, model_name=embedding_model)
        self.vector_store = VectorStore(embedder=self.embedder)

        self.pages = []
        self.logical_docs = []
        self.chunks = []

    def run(self, folder_path: str):
        """
        Full indexing pipeline:
        load PDFs -> classify pages -> build logical docs -> chunk -> embed -> index
        """
        print("Loading PDFs...", flush=True)
        self.pages = load_pdfs_from_folder(folder_path)

        print("Classifying doc types...", flush=True)
        self.pages = self.classifier.attach_doc_type_to_pages(self.pages)

        print("Building logical documents...", flush=True)
        self.logical_docs = self.logical_doc_builder.build_logical_documents(self.pages)

        print("Chunking logical documents...", flush=True)
        self.chunks = chunk_logical_documents(
            self.logical_docs,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )

        print("Building vector index...", flush=True)
        self.vector_store.build_index(self.chunks)

        print("Indexing complete.", flush=True)

        return {
            "pages": self.pages,
            "logical_docs": self.logical_docs,
            "chunks": self.chunks,
            "vector_store": self.vector_store,
            "stats": {
                "num_pages": len(self.pages),
                "num_logical_docs": len(self.logical_docs),
                "num_chunks": len(self.chunks),
                "embedder_model": self.embedder.get_model_name(),
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
            },
        }