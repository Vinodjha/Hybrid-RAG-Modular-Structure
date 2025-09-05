from fastapi import APIRouter
from app.store.memory import STATE

router = APIRouter(tags=["health"])

@router.get("/health")
def health():
    return {
        "ok" : True,
        "chunks": len(STATE.TEXTS),
        "dense_index_built":STATE.faiss_index is not None,
        "sparse_index_built": STATE.tfidf_matrix is not None,
        }


