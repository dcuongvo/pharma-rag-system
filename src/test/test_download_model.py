from sentence_transformers import SentenceTransformer, CrossEncoder
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

embed_model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    device=device,
    local_files_only=True
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device=device,
    local_files_only=True
)

print("Both models loaded from local cache successfully.")

