import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

def dense_pool(query: str, emb_model: SentenceTransformer, faiss_index, k: int) -> List[int]:
    q_vec = emb_model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
   
    # Searches the FAISS index for the k nearest neighbors to the query vector.
    # `D` contains the distances and `I` contains the indices of the neighbors.
        
    D, I = faiss_index.search(np.array([q_vec]), k=k)
    return I[0].tolist()

def sparse_pool(query: str, vectorizer, tfidf_mat, k: int) -> List[int]:
    q_sparse = vectorizer.transform([query])

    # Calculates the dot product between the entire TF-IDF matrix and the query's sparse vector.
    # This efficiently computes the relevance score for each document based on shared keywords.
    scores = (tfidf_mat @ q_sparse.T).toarray().ravel() # Convert to 1D array
    return np.argsort(-scores)[:k].tolist()

def merge_pools(dense_ids: List[int], sparse_ids: List[int], limit: int) -> List[int]:
    # `dict.fromkeys()` is a fast and concise way to remove duplicate indices while preserving order.
    return list(dict.fromkeys(dense_ids + sparse_ids))[:limit] # Keep only up to `limit` items