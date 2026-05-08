"""
LLM Integration Module for NeuralCanvas.
Uses Google Gemini API for topic content generation and relationship labeling.
Handles API errors gracefully with fallback mechanisms.
"""

import json
import google.generativeai as genai
from config import GEMINI_API_KEY

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


def _get_model():
    """Get configured Gemini model instance."""
    return genai.GenerativeModel('gemini-1.5-flash')


def generate_topic_content(topic):
    """
    Generate rich structured explanation for a given topic using Gemini.

    Prompt asks Gemini for definitions, key concepts, processes,
    relationships, examples, and applications (minimum 300 words).

    Args:
        topic (str): The topic name (e.g., "Neural Networks").

    Returns:
        str: The generated text response, or None if API fails.
    """
    try:
        model = _get_model()

        prompt = (
            f"Explain {topic} in detail. Include definitions, key concepts, "
            f"processes, relationships between concepts, examples, and applications. "
            f"Write in structured paragraphs, minimum 300 words. "
            f"Use clear, educational language suitable for students."
        )

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        print(f"[ERROR] Gemini API failed for topic generation: {e}")
        return None


def label_relationships(concept_pairs):
    """
    Use Gemini to label relationships between concept pairs.

    Sends a batch of concept pairs and asks for short relationship labels
    (3-5 words max) describing how concept A relates to concept B.

    Args:
        concept_pairs (list): List of dicts with 'source' and 'target' keys.
            Example: [{'source': 'neuron', 'target': 'neural network'}, ...]

    Returns:
        list: List of relationship labels (same order as input pairs).
            Returns generic labels if API fails (graceful fallback).
    """
    if not concept_pairs:
        return []

    # Fallback labels for when API fails
    fallback_labels = ["relates to"] * len(concept_pairs)

    try:
        model = _get_model()

        # Format pairs for the prompt
        pairs_text = "\n".join(
            [f'{i+1}. "{p["source"]}" → "{p["target"]}"'
             for i, p in enumerate(concept_pairs)]
        )

        prompt = (
            f"For each pair of concepts below, provide a short relationship label "
            f"(3-5 words max) describing how the first concept relates to the second. "
            f"Return ONLY a JSON array of strings, one label per pair, in order.\n\n"
            f"Concept pairs:\n{pairs_text}\n\n"
            f"Return format: [\"label1\", \"label2\", ...]\n"
            f"Return ONLY the JSON array, nothing else."
        )

        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # Clean up response — remove markdown code fences if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        # Parse JSON response
        labels = json.loads(response_text)

        if isinstance(labels, list) and len(labels) == len(concept_pairs):
            return labels
        elif isinstance(labels, list):
            # Pad or trim to match expected length
            while len(labels) < len(concept_pairs):
                labels.append("relates to")
            return labels[:len(concept_pairs)]
        else:
            return fallback_labels

    except json.JSONDecodeError as e:
        print(f"[WARNING] Failed to parse Gemini response as JSON: {e}")
        return fallback_labels

    except Exception as e:
        print(f"[WARNING] Gemini API failed for relationship labeling: {e}")
        return fallback_labels
