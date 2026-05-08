# 🧠 NeuralCanvas — AI Based Concept Map Learning System

NeuralCanvas is an AI-powered web application that transforms topics or raw text into interactive, visual concept maps. It combines **NLP**, **classical Machine Learning**, and **LLM integration** to extract, cluster, classify, and visualize knowledge.

---

## ✨ Features

- **Two Input Modes**: Topic Mode (Gemini generates content) and Text Mode (paste your own notes)
- **NLP Pipeline**: spaCy NER + TF-IDF keyword extraction + co-occurrence matrix + cosine similarity
- **ML Pipeline**: K-Means clustering, PCA dimensionality reduction, Random Forest concept classification
- **LLM Integration**: Google Gemini API for content generation and relationship labeling
- **Interactive Graph**: Cytoscape.js with draggable nodes, zoom, pan, and click-to-highlight
- **Export**: Download concept map as PNG

---

## 📁 Project Structure

```
neuralcanvas/
├── app.py                  # Flask app entry point
├── config.py               # Gemini API key config
├── requirements.txt        # Python dependencies
├── nlp/
│   ├── preprocessor.py     # Tokenization, stopwords, lemmatization
│   ├── extractor.py        # TF-IDF + NER keyword extraction
│   └── relations.py        # Co-occurrence + cosine similarity edges
├── ml/
│   ├── clustering.py       # K-Means clustering
│   ├── pca_reducer.py      # PCA 2D positioning
│   ├── classifier.py       # Random Forest concept classifier
│   └── saved/              # Saved ML models (.pkl)
├── llm/
│   └── gemini.py           # Gemini API integration
├── routes/
│   ├── generate.py         # /api/generate-from-topic & /api/generate-from-text
│   ├── classify.py         # /api/classify-concepts
│   └── clusters.py         # /api/clusters
├── static/
│   ├── css/style.css       # Dark theme styling
│   └── js/main.js          # Cytoscape.js rendering + API logic
└── templates/
    └── index.html          # Dashboard layout
```

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/H8rsh100/NeuralCanvas.git
cd NeuralCanvas
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download spaCy English model
```bash
python -m spacy download en_core_web_sm
```

### 5. Set Gemini API Key
Set your Google Gemini API key as an environment variable:

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY = "your-api-key-here"
```

**macOS/Linux:**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or edit `config.py` directly.

### 6. Run the application
```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## 🔌 API Endpoints

### `POST /api/generate-from-topic`
Generate a concept map from a topic name.
```json
{
    "topic": "Neural Networks"
}
```

### `POST /api/generate-from-text`
Generate a concept map from raw text/notes.
```json
{
    "text": "Your study notes here..."
}
```

### `POST /api/classify-concepts`
Classify concepts into categories using Random Forest.
```json
{
    "concepts": ["gradient descent", "neuron", "backpropagation"]
}
```
**Returns:** `{ "gradient descent": "Process", "neuron": "Definition", ... }`

### `POST /api/clusters`
Get K-Means cluster assignments for concepts.
```json
{
    "concepts": ["gradient descent", "neuron", "activation function"]
}
```

---

## 🎓 ML/AI Techniques Used

| Technique | Module | Description |
|-----------|--------|-------------|
| TF-IDF Vectorization | `nlp/extractor.py` | Extract important keywords by term frequency |
| Named Entity Recognition | `nlp/extractor.py` | Identify concepts using spaCy NER |
| Co-occurrence Matrix | `nlp/relations.py` | Build relationships from sentence co-occurrence |
| Cosine Similarity | `nlp/relations.py` | Measure concept relatedness |
| K-Means Clustering | `ml/clustering.py` | Group concepts into topic clusters |
| PCA | `ml/pca_reducer.py` | Reduce to 2D for map positioning |
| Random Forest | `ml/classifier.py` | Classify concepts into categories |
| Gemini LLM | `llm/gemini.py` | Generate content and label relationships |

---

## 📄 License

This project is for educational purposes.
