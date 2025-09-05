import os
import pickle
import faiss

def save_snapshot(index_dir:str, STATE) -> None:

    os.makedirs(index_dir, exist_ok=True)
    with open(os.path.join(index_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(STATE.METAS, f)

    if STATE.faiss_index is not None:
        faiss.write_index(STATE.faiss_index, os.path.join(index_dir, 'faiss.index'))
    
    if STATE.tfidf is not None:
        with open(os.path.join(index_dir, 'tfidf.pkl'), 'wb') as f:
            pickle.dump((STATE.tfidf, STATE.tfidf_matrix), f)

    if STATE.tfidf_mat is not None:
        with open(os.path.join(index_dir, 'tfidf_matrix.pkl'), 'wb') as f:
            pickle.dump(STATE.tfidf_matrix, f)

            