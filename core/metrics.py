import numpy as np
from itertools import combinations
from core.l2_norm_cosine_func import compute_similarity

def compute_far_frr(data, threshold=0.3, similarity_func=None):
    """
    Calculate FAR FRR rate

    Args:
        data: list of tuples (person, image_name, embedding)
        threshold:
        similarity_func: calculate cosine similarity
    Returns:
        far, frr
    """
    if similarity_func is None:
        similarity_func = compute_similarity

    genuine_sims = []
    impostor_sims = []

    for (p1, _, e1), (p2, _, e2) in combinations(data, 2):
        sim = similarity_func(e1, e2)
        if p1 == p2:
            genuine_sims.append(sim)
        else:
            impostor_sims.append(sim)

    genuine_sims = np.array(genuine_sims)
    impostor_sims = np.array(impostor_sims)

    frr = np.mean(genuine_sims < threshold) if len(genuine_sims) > 0 else 0.0
    far = np.mean(impostor_sims >= threshold) if len(impostor_sims) > 0 else 0.0

    return far, frr