from fastapi import APIRouter, UploadFile, HTTPException, File
from app.core.settings import settings
from app.ingestion.pdf import chunk_pdf
from app.indexing.dense import build_dense_index
from app.indexing.sparse import build_sparse_index
from app.indexing.persist import save_snapshot
from app.store.memory import STATE
from app.utils.hashing import hash_text
import os

router = APIRouter(tags=["index"])

@router.post("/index")
async def index(file: UploadFile):
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.INDEX_DIR, exist_ok=True)

    path = os.path.join(settings.DATA_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())

    items = chunk_pdf(path)  #[{"text"."page","source",...}..]

    for it in items:
        STATE.TEXTS.append(it["text"])
        STATE.METAS.append({
            "page": it.get("page",-1),
            "source": it.get("source",os.path.basename(path)),
            "id": hash_text(it["text"])
        })

    #(Re)build the indexes
    STATE.faiss_index = build_dense_index(STATE.TEXTS, STATE.emb_model)
    STATE.tfidf, STATE.tfidf_mat = build_sparse_index(STATE.TEXTS)
    
    #persist
    save_snapshot(settings.INDEX_DIR, STATE)

    return {"chunks_added": len(items), "total_chunks":len(STATE.TEXTS)}


