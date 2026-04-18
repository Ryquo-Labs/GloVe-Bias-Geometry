import json
import numpy as np
import os
import glob

def load_embeddings(data_dir: str):
    """
    Reads chunked words (.json) and vectors (.npy), and combines them into:
    - all_words: list mapping index -> word
    - word_to_index: dict mapping word -> index
    - vocab: set of all words
    - embeddings_matrix: 2D numpy array of shape (vocab_size, embedding_dim)
    """
    all_words = []
    vector_arrays = []
    
    # assuming we have chunks 1 to N
    chunk_files = sorted(glob.glob(os.path.join(data_dir, "*_words.json")))
    
    for word_file in chunk_files:
        # Load words
        with open(word_file, 'r', encoding='utf-8') as f:
            words = json.load(f)
            all_words.extend(words)
            
        # load corresponding numpy array
        npy_file = word_file.replace('_words.json', '_vectors.npy')
        if os.path.exists(npy_file):
            vectors = np.load(npy_file)
            vector_arrays.append(vectors)
            
    if vector_arrays:
        embeddings_matrix = np.concatenate(vector_arrays, axis=0)
    else:
        embeddings_matrix = np.array([])
        
    vocab = set(all_words)
    word_to_index = {w: i for i, w in enumerate(all_words)}
    
    return all_words, word_to_index, vocab, embeddings_matrix

