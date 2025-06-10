# vectorstore/embedder.py

import torch
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer

# Load model and determine device
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"

@lru_cache(maxsize=1000)
def get_cached_embedding(text: str):
    """Return cached embedding for a given text."""
    return tuple(embedding_model.encode([text], device=device)[0])

def embed_documents(texts: list[str]) -> list[np.ndarray]:
    """Compute embeddings for a list of texts."""
    return [np.array(get_cached_embedding(text)) for text in texts]
