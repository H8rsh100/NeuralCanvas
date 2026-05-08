"""
NLP Relations Module for NeuralCanvas.
Builds co-occurrence matrix and computes cosine similarity between concepts.
Returns filtered edges for the concept map graph.
"""

from sklearn.metrics.pairwise import cosine_similarity
from nlp.preprocessor import split_sentences
import numpy as np


def build_cooccurrence(concepts, raw_text):
    """
    Build a co-occurrence matrix: two concepts co-occur if they appear
    in the same sentence.

    Args:
        concepts (list): List of concept strings.
        raw_text (str): The original raw text.

    Returns:
        dict: Mapping of (concept_a, concept_b) -> co-occurrence count.
    """
    sentences = split_sentences(raw_text)
    cooccurrence = {}

    for sentence in sentences:
        sentence_lower = sentence.lower()
        # Find which concepts appear in this sentence
        present = [c for c in concepts if c.lower() in sentence_lower]

        # Count co-occurrences for all pairs in this sentence
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                pair = tuple(sorted([present[i], present[j]]))
                cooccurrence[pair] = cooccurrence.get(pair, 0) + 1

    return cooccurrence


def compute_concept_similarity(concepts, vectorizer, tfidf_matrix, feature_names):
    """
    Compute cosine similarity between all concept TF-IDF vectors.

    Args:
        concepts (list): List of concept strings.
        vectorizer: Fitted TfidfVectorizer instance.
        tfidf_matrix: The TF-IDF matrix from the extractor.
        feature_names (list): Feature names from the vectorizer.

    Returns:
        numpy.ndarray: Similarity matrix of shape (n_concepts, n_concepts).
        numpy.ndarray: Concept vectors of shape (n_concepts, n_features).
    """
    if vectorizer is None or tfidf_matrix is None:
        return np.array([]), np.array([])

    # Transform each concept into a TF-IDF vector using the same vectorizer
    concept_vectors = vectorizer.transform(concepts)

    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(concept_vectors)

    return similarity_matrix, concept_vectors.toarray()


def build_edges(concepts, raw_text, vectorizer, tfidf_matrix, feature_names, threshold=0.15):
    """
    Build edges between concepts using co-occurrence and cosine similarity.
    Only keep concept pairs where cosine similarity > threshold.

    Args:
        concepts (list): List of concept strings.
        raw_text (str): The original raw text.
        vectorizer: Fitted TfidfVectorizer instance.
        tfidf_matrix: The TF-IDF matrix.
        feature_names (list): Feature names from the vectorizer.
        threshold (float): Minimum cosine similarity to keep an edge.

    Returns:
        tuple: (edges, concept_vectors)
            - edges: list of dicts [{source, target, weight, similarity_score}]
            - concept_vectors: numpy array of concept vectors for ML pipeline
    """
    # Build co-occurrence counts
    cooccurrence = build_cooccurrence(concepts, raw_text)

    # Compute cosine similarity
    similarity_matrix, concept_vectors = compute_concept_similarity(
        concepts, vectorizer, tfidf_matrix, feature_names
    )

    edges = []

    if similarity_matrix.size == 0:
        # Fallback: use only co-occurrence if similarity can't be computed
        for (source, target), count in cooccurrence.items():
            edges.append({
                'source': source,
                'target': target,
                'weight': count,
                'similarity_score': 0.5  # Default similarity
            })
        return edges, concept_vectors

    # Build edges from similarity matrix, filtered by threshold
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            sim_score = float(similarity_matrix[i][j])

            if sim_score > threshold:
                pair = tuple(sorted([concepts[i], concepts[j]]))
                cooc_weight = cooccurrence.get(pair, 0)

                # Combined weight: similarity + co-occurrence bonus
                combined_weight = sim_score + (cooc_weight * 0.1)

                edges.append({
                    'source': concepts[i],
                    'target': concepts[j],
                    'weight': round(combined_weight, 4),
                    'similarity_score': round(sim_score, 4)
                })

    # Sort edges by weight descending
    edges.sort(key=lambda e: e['weight'], reverse=True)

    return edges, concept_vectors
