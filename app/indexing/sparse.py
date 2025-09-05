from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

def build_sparse_index(texts: List[str]) -> Tuple[TfidfVectorizer, any]:
    """Build a TF-IDF vectorizer and matrix from the provided texts."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix

