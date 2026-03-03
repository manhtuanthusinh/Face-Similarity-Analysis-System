import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# def normalize(vec):
#     """Chuẩn hóa vector embedding"""
#     return vec / (np.linalg.norm(vec) + 1e-10)

# def compute_similarity(emb1, emb2):
#     """Cosine similarity between 2 vector"""
#     return cosine_similarity([emb1], [emb2])[0][0]

def normalize(vec):
    """
    Normalize vector embedding (L2 normalize)
    """
    vec = np.asarray(vec, dtype=np.float32)
    return vec / (np.linalg.norm(vec) + 1e-10)

def compute_similarity(e1, e2):
    """
    Cosine similarity between 2 vector
    Args:
        e1, e2: numpy arrays
    Returns:
        float: similarity ∈ [-1, 1]
    """
    e1 = np.asarray(e1, dtype=np.float32)
    e2 = np.asarray(e2, dtype=np.float32)
    return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-10)