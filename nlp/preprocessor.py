"""
NLP Preprocessor Module for NeuralCanvas.
Handles text preprocessing: lowercasing, punctuation removal,
tokenization, stopword removal, and lemmatization using spaCy.
"""

import re
import spacy

# Load spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("[WARNING] spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    nlp = None


def preprocess_text(raw_text):
    """
    Accept raw text string as input and perform full preprocessing pipeline.

    Steps:
        1. Lowercase the text
        2. Remove punctuation and numbers
        3. Tokenize using spaCy
        4. Remove stopwords (spaCy's built-in stopword list)
        5. Lemmatize all tokens

    Args:
        raw_text (str): The raw input text to preprocess.

    Returns:
        dict: {
            'tokens': list of cleaned, lemmatized tokens,
            'cleaned_text': cleaned text string joined from tokens
        }
    """
    if not raw_text or not raw_text.strip():
        return {'tokens': [], 'cleaned_text': ''}

    if nlp is None:
        raise RuntimeError("spaCy model not loaded. Install it with: python -m spacy download en_core_web_sm")

    # Step 1: Lowercase
    text = raw_text.lower()

    # Step 2: Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', ' ', text)   # Remove punctuation
    text = re.sub(r'\d+', ' ', text)        # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace

    # Step 3: Tokenize using spaCy
    doc = nlp(text)

    # Step 4 & 5: Remove stopwords and lemmatize
    tokens = []
    for token in doc:
        # Skip stopwords, whitespace, and very short tokens
        if token.is_stop or token.is_space or len(token.text) < 2:
            continue
        # Lemmatize the token
        lemma = token.lemma_.strip()
        if lemma and len(lemma) > 1:
            tokens.append(lemma)

    # Build cleaned text from tokens
    cleaned_text = ' '.join(tokens)

    return {
        'tokens': tokens,
        'cleaned_text': cleaned_text
    }


def split_sentences(raw_text):
    """
    Split raw text into individual sentences using spaCy.

    Args:
        raw_text (str): The raw text to split.

    Returns:
        list: List of sentence strings.
    """
    if not raw_text or not raw_text.strip() or nlp is None:
        return []

    doc = nlp(raw_text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
