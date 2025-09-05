from typing import Any, Dict, List
from sentence_transformers import CrossEncoder

def rerank_and_pick(query: str, texts: List[str], metas: List[Dict[str, Any]], reranker: CrossEncoder, top_n: int) -> List[Dict[str, Any]]:
    if not texts:
        return []
    pairs = [(query, t) for t in texts]
    scores = reranker.predict(pairs)
    # Create a list of indices [0, 1, 2, ..., len(scores)-1]
    # Sort these indices by their score value (highest first, because reverse=True)
    # Keep only the top_n indices
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [{"text": texts[i], **metas[i]} for i in order]