# GloVe Bias Geometry Explorer

This Streamlit application allows you to explore pre-trained word embeddings visually and analytically. The app has two primary features:
1. **Analogy Equation Solver**: Solve semantic analogies by computing vectors like A - B + C.
2. **Bias Axis Projection**: Visualize how an arbitrary list of target words projects onto a 1D axis defined by two conceptually opposite terms.

## Assignment Instructions

This repository serves as a scaffolding activity for students. Several parts of the codebase have been replaced with `TODO` comments. Your task is to complete the implementation within the following files:

- **`utils/math.py`**: Implement vector math operations such as `cosine_similarity()`, `get_nearest_neighbors()`, and `calculate_projection()`.
- **`utils/text.py`**: Complete text utility functions including `sanitize_input()` and `validate_words()`.
- **`app.py`**: Fill in missing, straightforward Streamlit UI components based on the descriptive `TODO` block comments scattered throughout the file.

Follow the instructions written in the comments. Once everything is implemented correctly, the app should be able to visualize biases and solve embedding analogies!

## Setup & Installation

It is recommended to use your preferred environment. Simply install the required dependencies:

```bash
pip install -r requirements.txt
```

Note that embedding data already exists, and is placed in the `data/` folder. Embeddings are stored in chunks so that GitHub does not complain when we commit them.

## Running the Application

To run the Streamlit app locally, run:

```bash
streamlit run app.py
```

