from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer, CrossEncoder

class _Stats:

    #Text + Metadata
    TEXTS: List[str] = []
    METAS: List[Dict[str, Any]] = []

    #Indexes
    faiss_index = None
    tfidf = None
    tfidf_matrix = None

    #Models
    emb_model: Optional[SentenceTransformer] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2' )
    reranker: Optional[CrossEncoder] = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    #sessions: session_id -> deque[(q,a)]
    SESSIONS: dict[str,deque[tuple[str,str]]] = defaultdict(lambda: deque(maxlen=30)) 


STATE = _Stats()

