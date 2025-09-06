from fastapi import APIRouter
from app.store.memory import STATE

router = APIRouter(tags=["wipe"])


@router.post("/wipe", summary="Clear all in-memory chunks and indexes")
def wipe():
    STATE.TEXTS.clear()
    STATE.METAS.clear()
    STATE.faiss_index = None
    STATE.tfidf = None
    STATE.tfidf_mat = None
    return {"ok": True, "msg": "All chunks and indexes cleared", "chunks": 0}
