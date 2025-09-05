from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

def mmr_diversify(query: str, cand_texts: List[str], emb_model: SentenceTransformer, k: int = 8, lambda_mult: float = 0.5) -> List[int]:
    if not cand_texts:
        return []
    q_vec = emb_model.encode([query], normalize_embeddings=True)[0]
    cand_vecs = emb_model.encode(cand_texts, normalize_embeddings=True, show_progress_bar=False)
    sims = cand_vecs @ q_vec
    # Start with the single most relevant candidate (highest similarity to query)
    selected = [int(np.argmax(sims))]

    # Keep selecting until we have k items, or run out of candidates
    while len(selected) < min(k, len(cand_texts)):

        # Remaining candidates are those not yet selected
        remaining = [i for i in range(len(cand_texts)) if i not in selected]

        # Initialize best score and index for this iteration
        best_score, best_i = -1e9, None

        # Evaluate each remaining candidate
        for i in remaining:
            relevance = sims[i]  # similarity to query

            # Diversity term: max similarity to any already selected candidate
            diversity = max(float(cand_vecs[i] @ cand_vecs[j]) for j in selected)

            # Combine relevance and diversity into MMR score
            score = lambda_mult * relevance - (1 - lambda_mult) * diversity

            # Keep track of the best-scoring candidate
            if score > best_score:
                best_score, best_i = score, i

        # Add the chosen candidate to the selected set
        selected.append(best_i)

    # Return list of indices selected under MMR
    return selected
