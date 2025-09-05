from fastapi import APIRouter, HTTPException
from app.store.models import Query
from app.store.memory import STATE

from app.retrieval.search import dense_pool, sparse_pool, merge_pools
from app.retrieval.mmr import mmr_diversify
from app.retrieval.rerank import rerank_and_pick
from app.retrieval.selection import select_with_token_budget
from app.core.settings import settings
from app.core.budget import build_messages
from app.llm.groq_client import generate_answer

router = APIRouter(tags=["query"])
@router.post("/query")
def query(q: Query):
    if STATE.faiss_index is None or STATE.tfidf_mat is None:
        raise HTTPException(status_code=400, detail="Indexes are not built, upload a pdf")

    # A) Dense pool
    dense_ids = dense_pool(q.query, STATE.emb_model, STATE.faiss_index, k=max(1, settings.K_FETCH // 2))
    # B) Sparse pool
    sparse_ids = sparse_pool(q.query, STATE.tfidf, STATE.tfidf_mat, k=max(1, settings.K_FETCH // 2))
    # C) Merge & dedupe
    pool_ids = merge_pools(dense_ids, sparse_ids, limit=settings.K_FETCH)
    pool_texts = [STATE.TEXTS[i] for i in pool_ids]
    pool_metas = [STATE.METAS[i] for i in pool_ids]

    # D) MMR
    mmr_idx = mmr_diversify(q.query, pool_texts, STATE.emb_model, k=min(settings.K_MMR, len(pool_ids)))
    mmr_texts = [pool_texts[i] for i in mmr_idx]
    mmr_metas = [pool_metas[i] for i in mmr_idx]

    # E) Rerank -> top M_FINAL
    docs = rerank_and_pick(q.query, mmr_texts, mmr_metas, STATE.reranker, top_n=min(settings.M_FINAL, len(mmr_texts)))

    # F) Token budget selection
    docs_budgeted = select_with_token_budget(docs, settings.TOKEN_BUDGET)

    # G) Build messages & call LLM
    rag_texts = [f"[p{d['page']}] {d['text']}" for d in docs]
    history_pairs = list(STATE.SESSIONS[q.session_id])
    history_text = "\n".join(f"Q: {qq}\nA: {aa}" for qq, aa in history_pairs)
    messages, budgets, kept_idx, _ = build_messages(
        question=q.query,
        system=settings.SYSTEM_PROMPT,
        history=history_text,
        rag_chunks=rag_texts,
    )
    used_docs = [docs[i] for i in kept_idx]
    answer = generate_answer(messages, max_tokens=min(budgets["output"], q.max_answer_tokens or budgets["output"]))

    # Save session history
    STATE.SESSIONS[q.session_id].append((q.query, answer))

    # Minimal response
    retrieved = [{"page": d["page"], "source": d["source"], "preview": d["text"][:300].replace("\n", " ")} for d in docs_budgeted]
    return {"answer": answer, "retrieved_chunks": retrieved}

@router.post("/reset/{session_id}")
def reset_session(session_id: str):
    STATE.SESSIONS.pop(session_id, None)
    return {"ok": True}

