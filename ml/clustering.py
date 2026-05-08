"""
ML Clustering Module for NeuralCanvas.
Runs K-Means clustering on concept TF-IDF vectors to group related
concepts into topic clusters with color assignments.
"""

from sklearn.cluster import KMeans
import numpy as np

# Predefined palette of 8 distinct cluster colors
CLUSTER_COLORS = [
    '#38bdf8',  # Sky blue
    '#f472b6',  # Pink
    '#a78bfa',  # Purple
    '#34d399',  # Emerald
    '#fb923c',  # Orange
    '#facc15',  # Yellow
    '#f87171',  # Red
    '#22d3ee',  # Cyan
]


def perform_clustering(concept_vectors, concepts):
    """
    Run K-Means clustering on concept vectors.

    K is set to min(5, num_concepts // 2), with a minimum of 1.

    Args:
        concept_vectors (numpy.ndarray): TF-IDF vectors for each concept.
            Shape: (n_concepts, n_features)
        concepts (list): List of concept strings (same order as vectors).

    Returns:
        dict: {
            'cluster_assignments': dict mapping concept -> cluster_id,
            'cluster_colors': dict mapping concept -> hex color string,
            'k': number of clusters used,
            'labels': list of cluster labels (same order as concepts)
        }
    """
    n_concepts = len(concepts)

    if n_concepts == 0:
        return {
            'cluster_assignments': {},
            'cluster_colors': {},
            'k': 0,
            'labels': []
        }

    # Determine K
    k = min(5, max(1, n_concepts // 2))

    # Edge case: if only 1 concept, assign to cluster 0
    if n_concepts <= k:
        k = max(1, n_concepts)

    # Ensure concept_vectors is a numpy array
    if not isinstance(concept_vectors, np.ndarray):
        concept_vectors = np.array(concept_vectors)

    # Handle case where vectors might be empty or 1D
    if concept_vectors.ndim == 1:
        concept_vectors = concept_vectors.reshape(1, -1)

    # Run K-Means
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    labels = kmeans.fit_predict(concept_vectors)

    # Build mappings
    cluster_assignments = {}
    cluster_colors = {}

    for i, concept in enumerate(concepts):
        cluster_id = int(labels[i])
        cluster_assignments[concept] = cluster_id
        cluster_colors[concept] = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]

    return {
        'cluster_assignments': cluster_assignments,
        'cluster_colors': cluster_colors,
        'k': k,
        'labels': [int(l) for l in labels]
    }
