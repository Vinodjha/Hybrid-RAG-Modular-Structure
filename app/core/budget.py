import os
from typing import List, Tuple, Dict
from transformers import AutoTokenizer

from dotenv import load_dotenv

load_dotenv()

# ---- Budgets (env overrideable) ----
MAX_CTX = int(os.getenv("MAX_CTX_TOKENS", "131072"))     # Llama-3.3 on Groq
BUDGET_SYSTEM   = int(os.getenv("BUDGET_SYSTEM",   "300"))
BUDGET_QUESTION = int(os.getenv("BUDGET_QUESTION", "200"))
BUDGET_HISTORY  = int(os.getenv("BUDGET_HISTORY",  "2000"))
BUDGET_RAG      = int(os.getenv("BUDGET_RAG",      "6000"))  # set 10000 if you like
BUDGET_OUTPUT   = int(os.getenv("BUDGET_OUTPUT",   "900"))
HEADROOM        = int(os.getenv("BUDGET_HEADROOM", "1000"))


hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
_tok = AutoTokenizer.from_pretrained(
    os.getenv("TOKENIZER_ID", "meta-llama/Llama-3.1-8B-Instruct"),
    use_fast=True,
    token=hf_token,         # passes your HF token
)


def ntoks(s: str) -> int:
    if not s: return 0
    return len(_tok.encode(s, add_special_tokens=False))

def trim_to(s: str, budget: int) -> str:
    if ntoks(s) <= budget: return s or ""
    ids = _tok.encode(s or "", add_special_tokens=False)[:budget]
    return _tok.decode(ids)

def pack_chunks(chunks: List[str], budget: int) -> Tuple[str, List[int]]:
    kept, kept_idx, used = [], [], 0
    for i, ch in enumerate(chunks):
        t = ntoks(ch)
        if used + t > budget: continue
        kept.append(ch); kept_idx.append(i); used += t
    return "\n\n---\n\n".join(kept), kept_idx

def build_messages(
    question: str,
    system: str,
    history: str,
    rag_chunks: List[str]) -> Tuple[list, Dict[str, int], List[int], int]:
    """Return (messages, budgets, kept_idx, total_prompt_tokens)."""
    sys = trim_to(system,   BUDGET_SYSTEM)
    qst = trim_to(question, BUDGET_QUESTION)
    hst = trim_to(history or "", BUDGET_HISTORY)
    rag_text, kept_idx = pack_chunks(rag_chunks or [], BUDGET_RAG)

    user = (
        "QUESTION:\n" + qst +
        "\n\nCONTEXT (RAG):\n" + rag_text +
        "\n\nHISTORY (summary):\n" + hst +
        "\n\nInstructions: Answer using CONTEXT. If unsure, say so. Include citations [pX]."
    )

    total_in = ntoks(sys) + ntoks(user)
    limit = MAX_CTX - BUDGET_OUTPUT - HEADROOM
    if total_in > limit and rag_chunks:
        shrink = max(1000, BUDGET_RAG - (total_in - limit))
        rag_text, kept_idx = pack_chunks(rag_chunks, shrink)
        user = (
            "QUESTION:\n" + qst +
            "\n\nCONTEXT (RAG):\n" + rag_text +
            "\n\nHISTORY (summary):\n" + hst +
            "\n\nInstructions: Answer using CONTEXT. If unsure, say so. Include citations [pX]."
        )
        total_in = ntoks(sys) + ntoks(user)

    messages = [
        {"role": "system", "content": sys},
        {"role": "user",   "content": user},
    ]
    budgets = dict(system=BUDGET_SYSTEM, question=BUDGET_QUESTION, history=BUDGET_HISTORY,
                   rag=BUDGET_RAG, output=BUDGET_OUTPUT, headroom=HEADROOM, max_ctx=MAX_CTX)
    return messages, budgets, kept_idx, total_in
