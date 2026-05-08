"""
Clusters Route for NeuralCanvas.
Handles /api/clusters endpoint.
Returns K-Means cluster assignments for given concepts.
"""

from flask import Blueprint, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from ml.clustering import perform_clustering

clusters_bp = Blueprint('clusters', __name__)


@clusters_bp.route('/clusters', methods=['POST'])
def get_clusters():
    """
    POST /api/clusters
    Request body: { "concepts": ["gradient descent", "neuron", ...] }

    Returns K-Means cluster assignments for the given concepts.
    Vectorizes concepts internally using TF-IDF before clustering.
    """
    data = request.get_json()

    if not data or 'concepts' not in data:
        return jsonify({'error': 'Missing "concepts" field in request body.'}), 400

    concepts = data['concepts']

    if not isinstance(concepts, list) or len(concepts) == 0:
        return jsonify({'error': '"concepts" must be a non-empty list of strings.'}), 400

    # Clean concepts
    concepts = [str(c).strip() for c in concepts if str(c).strip()]

    if not concepts:
        return jsonify({'error': 'No valid concepts provided.'}), 400

    # Vectorize concepts using TF-IDF
    vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 2), min_df=1)
    concept_vectors = vectorizer.fit_transform(concepts).toarray()

    # Run K-Means clustering
    result = perform_clustering(concept_vectors, concepts)

    return jsonify({
        'cluster_assignments': result['cluster_assignments'],
        'cluster_colors': result['cluster_colors'],
        'k': result['k']
    })
