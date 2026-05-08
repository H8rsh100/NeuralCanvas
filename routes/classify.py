"""
Classify Route for NeuralCanvas.
Handles /api/classify-concepts endpoint.
Returns concept categories via Random Forest classifier.
"""

from flask import Blueprint, request, jsonify
from ml.classifier import get_classifier

classify_bp = Blueprint('classify', __name__)


@classify_bp.route('/classify-concepts', methods=['POST'])
def classify_concepts():
    """
    POST /api/classify-concepts
    Request body: { "concepts": ["gradient descent", "neuron", ...] }

    Returns: { "gradient descent": "Process", "neuron": "Definition", ... }
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

    # Classify using Random Forest
    classifier = get_classifier()
    categories = classifier.classify(concepts)

    return jsonify(categories)
