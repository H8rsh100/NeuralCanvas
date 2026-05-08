"""
NLP Concept Extractor Module for NeuralCanvas.
Extracts key concepts using spaCy NER and scikit-learn TF-IDF vectorization.
Returns deduplicated concept list with importance scores and the TF-IDF matrix.
"""

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from nlp.preprocessor import preprocess_text, split_sentences

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("[WARNING] spaCy model 'en_core_web_sm' not found.")
    nlp = None


def extract_ner_entities(raw_text):
    """
    Run spaCy NER on raw text to extract named entities tagged as concepts.

    Args:
        raw_text (str): The raw input text.

    Returns:
        list: List of unique entity strings.
    """
    if nlp is None or not raw_text:
        return []

    doc = nlp(raw_text)
    entities = set()
    for ent in doc.ents:
        cleaned = ent.text.strip().lower()
        if len(cleaned) > 1:
            entities.add(cleaned)
    return list(entities)


def extract_tfidf_keywords(raw_text, top_n=20):
    """
    Run TF-IDF on sentences to extract top N keywords by TF-IDF score.

    Args:
        raw_text (str): The raw input text.
        top_n (int): Number of top keywords to extract.

    Returns:
        tuple: (keywords_with_scores, tfidf_matrix, feature_names, vectorizer)
            - keywords_with_scores: list of (keyword, score) tuples
            - tfidf_matrix: sparse TF-IDF matrix
            - feature_names: list of all feature names from vectorizer
            - vectorizer: fitted TfidfVectorizer instance
    """
    sentences = split_sentences(raw_text)
    if not sentences:
        return [], None, [], None

    # Preprocess each sentence
    processed_sentences = []
    for sent in sentences:
        result = preprocess_text(sent)
        if result['cleaned_text']:
            processed_sentences.append(result['cleaned_text'])

    if not processed_sentences:
        return [], None, [], None

    # Fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )

    tfidf_matrix = vectorizer.fit_transform(processed_sentences)
    feature_names = vectorizer.get_feature_names_out()

    # Compute average TF-IDF score per feature across all sentences
    avg_scores = tfidf_matrix.mean(axis=0).A1  # Convert matrix to 1D array

    # Get top N keywords by average score
    scored = list(zip(feature_names, avg_scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_keywords = scored[:top_n]

    return top_keywords, tfidf_matrix, feature_names, vectorizer


def extract_concepts(raw_text, top_n=20):
    """
    Full concept extraction pipeline: NER + TF-IDF, deduplicated.

    Args:
        raw_text (str): The raw input text.
        top_n (int): Max number of concepts to extract.

    Returns:
        dict: {
            'concepts': list of concept strings,
            'scores': dict mapping concept -> TF-IDF weight score,
            'tfidf_matrix': the TF-IDF matrix (for ML pipeline),
            'feature_names': list of feature names,
            'vectorizer': fitted TfidfVectorizer instance
        }
    """
    # Extract NER entities
    ner_entities = extract_ner_entities(raw_text)

    # Extract TF-IDF keywords
    tfidf_keywords, tfidf_matrix, feature_names, vectorizer = extract_tfidf_keywords(raw_text, top_n)

    # Merge and deduplicate
    concept_scores = {}

    # Add NER entities with a base score
    for entity in ner_entities:
        concept_scores[entity] = concept_scores.get(entity, 0) + 0.3

    # Add TF-IDF keywords with their scores
    for keyword, score in tfidf_keywords:
        keyword_lower = keyword.lower()
        concept_scores[keyword_lower] = concept_scores.get(keyword_lower, 0) + score

    # Sort by score and take top N
    sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
    top_concepts = sorted_concepts[:top_n]

    concepts = [c[0] for c in top_concepts]
    scores = {c[0]: round(c[1], 4) for c in top_concepts}

    return {
        'concepts': concepts,
        'scores': scores,
        'tfidf_matrix': tfidf_matrix,
        'feature_names': list(feature_names) if feature_names is not None else [],
        'vectorizer': vectorizer
    }
