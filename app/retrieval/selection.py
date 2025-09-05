from typing import Any, Dict, List

def estimate_tokens(text: str) -> int:
    return len(text.split())

def select_with_token_budget(docs: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
    total, out = 0, []
    for d in docs:
        t = estimate_tokens(d["text"])
        if total + t <= budget:
            out.append(d); total += t
        else:
            break
    return out