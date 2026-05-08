"""
ML PCA Dimensionality Reduction Module for NeuralCanvas.
Reduces TF-IDF concept vectors to 2D coordinates for spatial
positioning on the concept map canvas.
"""

from sklearn.decomposition import PCA
import numpy as np


def reduce_dimensions(concept_vectors, concepts, canvas_width=800, canvas_height=600):
    """
    Run PCA (n_components=2) to reduce concept vectors to 2D coordinates.
    Scales coordinates to fit within the specified canvas dimensions.

    Args:
        concept_vectors (numpy.ndarray): TF-IDF vectors for each concept.
            Shape: (n_concepts, n_features)
        concepts (list): List of concept strings (same order as vectors).
        canvas_width (int): Width of the canvas to scale coordinates to.
        canvas_height (int): Height of the canvas to scale coordinates to.

    Returns:
        dict: {
            'positions': dict mapping concept -> {'x': float, 'y': float},
            'raw_coords': numpy.ndarray of shape (n_concepts, 2)
        }
    """
    n_concepts = len(concepts)

    if n_concepts == 0:
        return {'positions': {}, 'raw_coords': np.array([])}

    # Ensure concept_vectors is a numpy array
    if not isinstance(concept_vectors, np.ndarray):
        concept_vectors = np.array(concept_vectors)

    if concept_vectors.ndim == 1:
        concept_vectors = concept_vectors.reshape(1, -1)

    # Handle edge case: if only 1 concept, place at center
    if n_concepts == 1:
        positions = {concepts[0]: {'x': canvas_width / 2, 'y': canvas_height / 2}}
        return {'positions': positions, 'raw_coords': np.array([[0.0, 0.0]])}

    # Determine n_components (can't exceed n_samples or n_features)
    n_components = min(2, n_concepts, concept_vectors.shape[1])

    # Run PCA
    pca = PCA(n_components=n_components, random_state=42)
    coords_2d = pca.fit_transform(concept_vectors)

    # If PCA only gave 1 component, add a zero second dimension
    if coords_2d.shape[1] == 1:
        coords_2d = np.column_stack([coords_2d, np.zeros(n_concepts)])

    # Scale coordinates to fit within canvas with padding
    padding = 60  # Pixels of padding from edges

    x_min, x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
    y_min, y_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()

    # Avoid division by zero
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0

    # Normalize to [0, 1] then scale to canvas with padding
    scaled_coords = np.zeros_like(coords_2d)
    scaled_coords[:, 0] = ((coords_2d[:, 0] - x_min) / x_range) * (canvas_width - 2 * padding) + padding
    scaled_coords[:, 1] = ((coords_2d[:, 1] - y_min) / y_range) * (canvas_height - 2 * padding) + padding

    # Build positions mapping
    positions = {}
    for i, concept in enumerate(concepts):
        positions[concept] = {
            'x': round(float(scaled_coords[i, 0]), 2),
            'y': round(float(scaled_coords[i, 1]), 2)
        }

    return {
        'positions': positions,
        'raw_coords': coords_2d
    }
