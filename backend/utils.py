import math
import numpy as np

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0: return 1.0
    return 1.0 - float(np.dot(a, b) / denom)

def top_k_neighbors(query: np.ndarray, corpus: np.ndarray, k=3, metric="cosine"):
    """Brute force, matches sklearn KNN for verification."""
    dists = []
    for i, row in enumerate(corpus):
        d = cosine_distance(query, row) if metric=="cosine" else float(np.linalg.norm(query-row))
        dists.append((d,i))
    dists.sort(key=lambda x: x[0])
    return dists[:k]
