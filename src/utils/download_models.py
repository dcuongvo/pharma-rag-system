from sentence_transformers import SentenceTransformer, CrossEncoder
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

print("Downloading embedding model...")
embed_model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    device=device
)

print("Downloading reranker model...")
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device=device
)

print("Done. Models should now be cached locally.")