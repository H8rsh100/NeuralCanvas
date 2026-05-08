"""
Generate Routes for NeuralCanvas.
Handles /api/generate-from-topic and /api/generate-from-text endpoints.
Orchestrates the full NLP → ML → LLM pipeline to produce graph JSON.
"""

from flask import Blueprint, request, jsonify
from nlp.extractor import extract_concepts
from nlp.relations import build_edges
from ml.clustering import perform_clustering
from ml.pca_reducer import reduce_dimensions
from ml.classifier import get_classifier
from llm.gemini import generate_topic_content, label_relationships

generate_bp = Blueprint('generate', __name__)


def _build_graph(raw_text, use_gemini_labels=True):
    """
    Internal function: run the full NLP + ML pipeline on text to produce graph JSON.

    Args:
        raw_text (str): The text to process.
        use_gemini_labels (bool): Whether to use Gemini for edge labels.

    Returns:
        dict: { 'nodes': [...], 'edges': [...] }
    """
    # Step 1: Extract concepts via NLP pipeline
    extraction = extract_concepts(raw_text, top_n=20)
    concepts = extraction['concepts']
    scores = extraction['scores']
    vectorizer = extraction['vectorizer']
    tfidf_matrix = extraction['tfidf_matrix']
    feature_names = extraction['feature_names']

    if not concepts:
        return {'nodes': [], 'edges': [], 'error': 'No concepts could be extracted from the text.'}

    # Step 2: Build edges (co-occurrence + cosine similarity)
    edges, concept_vectors = build_edges(
        concepts, raw_text, vectorizer, tfidf_matrix, feature_names
    )

    # Step 3: ML — K-Means clustering
    if concept_vectors is not None and len(concept_vectors) > 0:
        clustering = perform_clustering(concept_vectors, concepts)
    else:
        # Fallback: use vectorizer to get vectors
        fallback_vectors = vectorizer.transform(concepts).toarray()
        clustering = perform_clustering(fallback_vectors, concepts)
        concept_vectors = fallback_vectors

    # Step 4: ML — PCA dimensionality reduction for 2D positioning
    if concept_vectors is not None and len(concept_vectors) > 0:
        pca_result = reduce_dimensions(concept_vectors, concepts)
    else:
        fallback_vectors = vectorizer.transform(concepts).toarray()
        pca_result = reduce_dimensions(fallback_vectors, concepts)

    positions = pca_result['positions']

    # Step 5: ML — Random Forest classification
    classifier = get_classifier()
    categories = classifier.classify(concepts)

    # Step 6: LLM — Label edges with Gemini (if enabled)
    edge_labels = []
    if use_gemini_labels and edges:
        concept_pairs = [{'source': e['source'], 'target': e['target']} for e in edges]
        edge_labels = label_relationships(concept_pairs)

    # Build nodes JSON
    nodes = []
    for concept in concepts:
        pos = positions.get(concept, {'x': 400, 'y': 300})
        nodes.append({
            'id': concept,
            'label': concept.title(),
            'cluster': clustering['cluster_assignments'].get(concept, 0),
            'color': clustering['cluster_colors'].get(concept, '#38bdf8'),
            'x': pos['x'],
            'y': pos['y'],
            'category': categories.get(concept, 'Definition'),
            'score': scores.get(concept, 0.0)
        })

    # Build edges JSON
    edges_json = []
    for i, edge in enumerate(edges):
        label = edge_labels[i] if i < len(edge_labels) else 'relates to'
        edges_json.append({
            'source': edge['source'],
            'target': edge['target'],
            'weight': edge['weight'],
            'label': label
        })

    return {
        'nodes': nodes,
        'edges': edges_json
    }


@generate_bp.route('/generate-from-topic', methods=['POST'])
def generate_from_topic():
    """
    POST /api/generate-from-topic
    Request body: { "topic": "Neural Networks" }

    Calls Gemini → NLP → ML → graph JSON.
    """
    data = request.get_json()

    if not data or 'topic' not in data:
        return jsonify({'error': 'Missing "topic" field in request body.'}), 400

    topic = data['topic'].strip()
    if not topic:
        return jsonify({'error': 'Topic cannot be empty.'}), 400

    # Step 1: Generate content from Gemini
    raw_text = generate_topic_content(topic)

    if not raw_text:
        # Fallback: use topic name as minimal text
        raw_text = (
            f"{topic} is an important concept. {topic} involves various processes "
            f"and applications. The study of {topic} includes definitions, theories, "
            f"examples, and practical implementations."
        )

    # Step 2-6: Build graph
    graph = _build_graph(raw_text, use_gemini_labels=True)

    return jsonify(graph)


@generate_bp.route('/generate-from-text', methods=['POST'])
def generate_from_text():
    """
    POST /api/generate-from-text
    Request body: { "text": "...raw student notes..." }

    Raw text → NLP → ML → graph JSON. No Gemini for content generation.
    Gemini is still used for edge labeling (optional).
    """
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field in request body.'}), 400

    raw_text = data['text'].strip()
    if not raw_text:
        return jsonify({'error': 'Text cannot be empty.'}), 400

    if len(raw_text) < 20:
        return jsonify({'error': 'Text is too short. Please provide more detailed content.'}), 400

    # Build graph (use Gemini only for edge labels, not content generation)
    graph = _build_graph(raw_text, use_gemini_labels=True)

    return jsonify(graph)
