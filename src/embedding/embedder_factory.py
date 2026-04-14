from src.embedding.bge_embedder import BGEEmbedder
from src.utils.config import Settings


def get_embedder(embedder_type: str | None = None, model_name: str | None = None):
    embedder_type = embedder_type or Settings.EMBEDDER_PROVIDER
    model_name = model_name or Settings.EMBEDDING_MODEL

    if embedder_type == "bge":
        return BGEEmbedder(model_name=model_name)

    raise ValueError(f"Unsupported embedder type: {embedder_type}")