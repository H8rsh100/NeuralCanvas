"""
ML Concept Classifier Module for NeuralCanvas.
Trains a Random Forest Classifier on a built-in labeled dataset to
classify concepts into categories: Definition, Process, Example, Theory, Application.
Saves/loads trained model from ml/saved/rf_classifier.pkl.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Path to saved model
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved')
MODEL_PATH = os.path.join(MODEL_DIR, 'rf_classifier.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'rf_vectorizer.pkl')

# Categories
CATEGORIES = ["Definition", "Process", "Example", "Theory", "Application"]

# Built-in labeled training data (40+ samples covering all 5 categories)
TRAINING_DATA = [
    # Definition (8 samples)
    ("algorithm is a step by step procedure", "Definition"),
    ("variable is a named storage location", "Definition"),
    ("function is a reusable block of code", "Definition"),
    ("neural network is a computational model", "Definition"),
    ("database is an organized collection of data", "Definition"),
    ("protocol is a set of communication rules", "Definition"),
    ("class is a blueprint for creating objects", "Definition"),
    ("entropy is a measure of disorder or randomness", "Definition"),

    # Process (8 samples)
    ("training the model with gradient descent", "Process"),
    ("compiling source code into machine language", "Process"),
    ("sorting the array using quicksort method", "Process"),
    ("propagating errors backward through layers", "Process"),
    ("tokenizing text into individual words", "Process"),
    ("normalizing data to standard scale", "Process"),
    ("clustering data points into groups", "Process"),
    ("optimizing weights using backpropagation", "Process"),

    # Example (8 samples)
    ("for instance image recognition in self driving cars", "Example"),
    ("such as spam filtering in email systems", "Example"),
    ("for example linear regression predicts house prices", "Example"),
    ("consider the case of sentiment analysis on tweets", "Example"),
    ("like using decision trees for classification tasks", "Example"),
    ("an illustration of transfer learning in medical imaging", "Example"),
    ("a practical case of chatbot using natural language", "Example"),
    ("demonstrated by recommendation systems on netflix", "Example"),

    # Theory (8 samples)
    ("bayes theorem describes probability of an event", "Theory"),
    ("information theory studies quantification of information", "Theory"),
    ("universal approximation theorem states neural networks", "Theory"),
    ("central limit theorem describes distribution of means", "Theory"),
    ("graph theory studies mathematical structures of networks", "Theory"),
    ("complexity theory classifies computational problems", "Theory"),
    ("game theory analyzes strategic interactions between agents", "Theory"),
    ("probability theory provides foundation for statistics", "Theory"),

    # Application (8 samples)
    ("used in medical diagnosis prediction systems", "Application"),
    ("applied to autonomous vehicle navigation", "Application"),
    ("implemented in fraud detection banking systems", "Application"),
    ("deployed for real time language translation", "Application"),
    ("utilized in weather forecasting models", "Application"),
    ("powering search engine ranking algorithms", "Application"),
    ("enabling speech recognition in virtual assistants", "Application"),
    ("supporting drug discovery in pharmaceutical research", "Application"),
]


class ConceptClassifier:
    """Random Forest based concept classifier with auto-training."""

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self._load_or_train()

    def _load_or_train(self):
        """Load saved model or train a new one."""
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.vectorizer = joblib.load(VECTORIZER_PATH)
                print("[INFO] Loaded saved Random Forest classifier.")
                return
            except Exception as e:
                print(f"[WARNING] Failed to load saved model: {e}")

        # Train new model
        self._train()

    def _train(self):
        """Train the Random Forest classifier on built-in data."""
        print("[INFO] Training Random Forest classifier...")

        texts = [item[0] for item in TRAINING_DATA]
        labels = [item[1] for item in TRAINING_DATA]

        # Fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=200,
            ngram_range=(1, 2),
            min_df=1
        )
        X = self.vectorizer.fit_transform(texts)

        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.model.fit(X, labels)

        # Save model
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.vectorizer, VECTORIZER_PATH)
        print(f"[INFO] Random Forest classifier trained and saved to {MODEL_PATH}")

    def classify(self, concepts):
        """
        Classify a list of concepts into categories.

        Args:
            concepts (list): List of concept strings.

        Returns:
            dict: Mapping of concept -> category string.
        """
        if not concepts or self.model is None:
            return {}

        # Vectorize concepts
        X = self.vectorizer.transform(concepts)

        # Predict categories
        predictions = self.model.predict(X)

        return {concept: category for concept, category in zip(concepts, predictions)}

    def classify_with_confidence(self, concepts):
        """
        Classify concepts with confidence scores.

        Args:
            concepts (list): List of concept strings.

        Returns:
            dict: Mapping of concept -> {'category': str, 'confidence': float}
        """
        if not concepts or self.model is None:
            return {}

        X = self.vectorizer.transform(concepts)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        result = {}
        for i, concept in enumerate(concepts):
            max_prob = float(np.max(probabilities[i]))
            result[concept] = {
                'category': predictions[i],
                'confidence': round(max_prob, 4)
            }

        return result


# Singleton instance
_classifier_instance = None


def get_classifier():
    """Get or create the singleton classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ConceptClassifier()
    return _classifier_instance
