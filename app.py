import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from utils.data import load_embeddings
from utils.text import sanitize_input, validate_words
from utils.math import get_nearest_neighbors, calculate_projection

# -----------------------------------------------------------------
# App Configuration
# -----------------------------------------------------------------
st.set_page_config(
    page_title="GloVe Embedding Explorer",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------------------------------------------
# State Management
# -----------------------------------------------------------------
@st.cache_resource
def get_model():
    """Load model once into memory."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    all_words, word_to_index, vocab, embeddings_matrix = load_embeddings(data_dir)
    return all_words, word_to_index, vocab, embeddings_matrix

# Load embeddings
with st.spinner("Loading word embeddings from chunks..."):
    all_words, word_to_index, vocab, embeddings_matrix = get_model()

# -----------------------------------------------------------------
# Title
# -----------------------------------------------------------------
st.title("GloVe Embedding Explorer 🧠")
st.markdown("Interact with pre-trained GloVe word embeddings via analogy math or bias projections.")

st.divider()

# -----------------------------------------------------------------
# Feature 1: Analogy Equation Solver
# -----------------------------------------------------------------
st.header("1. Analogy Equation Solver")
st.write("Compute **A - B + C** to find related semantic concepts.")

col1, col_minus, col2, col_plus, col3 = st.columns([2, 0.5, 2, 0.5, 2])

with col1:
    word_a_input = st.text_input("Word A", placeholder="King")
with col_minus:
    st.markdown("<div style='text-align: center; font-size: 28px; font-weight: bold; margin-top: 28px;'>−</div>", unsafe_allow_html=True)
with col2:
    word_b_input = st.text_input("Word B", placeholder="Man")
with col_plus:
    st.markdown("<div style='text-align: center; font-size: 28px; font-weight: bold; margin-top: 28px;'>+</div>", unsafe_allow_html=True)
with col3:
    word_c_input = st.text_input("Word C", placeholder="Woman")

if st.button("Compute Analogy", type="primary"):
    word_a = sanitize_input(word_a_input)
    word_b = sanitize_input(word_b_input)
    word_c = sanitize_input(word_c_input)
    
    if not all([word_a, word_b, word_c]):
        st.warning("Please provide all three words.")
    else:
        valid_words, invalid_words = validate_words([word_a, word_b, word_c], vocab)
        
        if invalid_words:
            st.error(f"The following words are not in the dictionary: {', '.join(invalid_words)}")
        else:
            # Vector math
            v_a = embeddings_matrix[word_to_index[word_a]]
            v_b = embeddings_matrix[word_to_index[word_b]]
            v_c = embeddings_matrix[word_to_index[word_c]]
            
            v_target = v_a - v_b + v_c
            
            # Find nearest neighbors
            st.subheader(f"Nearest to: {word_a} - {word_b} + {word_c}")
            results = get_nearest_neighbors(v_target, embeddings_matrix, all_words, exclude_words={word_a, word_b, word_c}, topn=10)
            
            # Show output
            df = pd.DataFrame(results, columns=["Word", "Cosine Similarity"])
            df.index = np.arange(1, len(df) + 1)
            st.dataframe(df.style.format({"Cosine Similarity": "{:.4f}"}), width='stretch')

st.divider()

# -----------------------------------------------------------------
# Feature 2: Bias Axis Projection
# -----------------------------------------------------------------
st.header("2. Bias Axis Projection")
st.write("Project words onto a 1D semantic axis defined by conceptual opposites.")

axis_options = {
    "Man vs Woman": ("man", "woman"),
    "Rich vs Poor": ("rich", "poor"),
    "Good vs Bad": ("good", "bad"),
    "Science vs Arts": ("science", "arts"),
    "Old vs Young": ("old", "young")
}

selected_axis = st.selectbox("Select Axis (Pole X vs Pole Y)", list(axis_options.keys()))
target_words_input = st.text_area(
    "Target Words (comma separated)", 
    "doctor, nurse, teacher, boss, scientist, artist, creative, poet, math, actor, analyze, singer, entrepreneur, researcher, uncle, princess",
    help="Enter a comma-delimited list of target words to project onto the axis."
)

if st.button("Project Words", type="primary"):
    pole_x, pole_y = axis_options[selected_axis]
    pole_x = sanitize_input(pole_x)
    pole_y = sanitize_input(pole_y)
    
    target_words = [sanitize_input(w.strip()) for w in target_words_input.split(',')]
    target_words = [w for w in target_words if w] # remove empty
    
    # Needs to validate both pole words and target words
    valid_poles, invalid_poles = validate_words([pole_x, pole_y], vocab)
    if invalid_poles:
        st.error(f"The axis words {invalid_poles} are missing from the dictionary.")
    else:
        valid_targets, invalid_targets = validate_words(target_words, vocab)
        
        if invalid_targets:
            st.warning(f"Skipped these words as they are not in the dictionary: {', '.join(invalid_targets)}")
        
        if not valid_targets:
            st.error("No valid target words found in the model dictionary.")
        else:
            v_x = embeddings_matrix[word_to_index[pole_x]]
            v_y = embeddings_matrix[word_to_index[pole_y]]
            
            # Calculate projections
            scores = []
            for word in valid_targets:
                v_w = embeddings_matrix[word_to_index[word]]
                score = calculate_projection(v_w, v_x, v_y)
                scores.append((word, score))

            # Render plot
            fig, ax = plt.subplots()
            ax.get_yaxis().set_visible(False)
            ax.set_xlim(-1.05, 1.05)

            positions = [s[1] for s in scores]
            words = [s[0] for s in scores]

            if positions:
                # Sort positions and words by position
                positions, words = zip(*sorted(zip(positions, words)))

                # Calculate offsets for each word with DP
                # We love Leetcode!
                default_offset = 0.2
                offsets = [default_offset]
                for i in range(1, len(positions)):
                    if positions[i] - positions[i-1] < 0.05:
                        offsets.append(offsets[-1] + 0.25)
                    else:
                        offsets.append(default_offset)

                # Plot each word with an arrow pointing to its specified position
                for position, word, offset in zip(positions, words, offsets):
                    ax.annotate(word, xy=(position, 0), xytext=(position, offset),
                                arrowprops=dict(facecolor='black', arrowstyle="->", connectionstyle="arc3"),
                                ha='center', rotation=90)
                                
                # Provide enough unbounded scale factor explicitly mapped for Streamlit
                max_offset = max(offsets)
                ax.set_ylim(-0.4, max_offset + max_offset * 0.1 + 0.5) 
                fig.set_figheight(max(4, 2 + max_offset * 2.5))
                fig.set_figwidth(10)

            # Add a line at y=0
            ax.axhline(0, color='black')

            # Remove top, left, and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)    
            ax.spines['bottom'].set_position('zero')
            
            # Label axes clearly with optimal vertical margin
            ax.text(1.0, -0.2, pole_x.capitalize(), ha='right', va='top', fontsize=18, fontweight='bold')
            ax.text(-1.0, -0.2, pole_y.capitalize(), ha='left', va='top', fontsize=18, fontweight='bold')
            
            st.pyplot(fig)

