import numpy as np

def cosine_similarity(v, m):
    """
    Compute cosine similarity between vector v and matrix m.
    v: 1D array
    m: 2D array
    Returns: 1D array of similarities
    """
    v_norm = np.linalg.norm(v)
    m_norm = np.linalg.norm(m, axis=1)

    # Prevent division by zero
    v_norm = v_norm if v_norm > 0 else 1e-10
    m_norm = np.where(m_norm == 0, 1e-10, m_norm)
    
    return np.dot(m, v) / (v_norm * m_norm)

def get_nearest_neighbors(target_vector, embeddings_matrix, index_to_word, exclude_words=None, topn=10):
    """
    Finds the nearest neighbors for a target vector.
    """
    if exclude_words is None:
        exclude_words = set()
        
    sims = cosine_similarity(target_vector, embeddings_matrix)

    # Sort descending
    best_indices = np.argsort(sims)[::-1]
    
    results = []
    for idx in best_indices:
        word = index_to_word[idx]
        if word not in exclude_words:
            results.append((word, float(sims[idx])))
        if len(results) == topn:
            break
            
    return results

def calculate_projection(word_vec, pole1_vec, pole2_vec):
    """
    Project word_vec onto the axis defined by (pole1_vec - pole2_vec).
    Formula: (W . D) / (||W|| * ||D||) 
    where D = pole1_vec - pole2_vec
    """
    axis_vector = pole1_vec - pole2_vec
    w_norm = np.linalg.norm(word_vec)
    d_norm = np.linalg.norm(axis_vector)
    
    if w_norm == 0 or d_norm == 0:
        return 0.0
        
    score = np.dot(word_vec, axis_vector) / (w_norm * d_norm)
    return float(score)
