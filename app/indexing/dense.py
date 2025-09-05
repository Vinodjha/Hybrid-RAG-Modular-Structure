import numpy as np
import faiss 
from typing import List
from sentence_transformers import SentenceTransformer

def build_dense_index(texts: List[str], emb_model:SentenceTransformer):
    """Build a FAISS index from the provided texts using the given embedding model."""
    embeddings = emb_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings, dtype='float32'))
    return index