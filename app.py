"""
NeuralCanvas — AI Based Concept Map Learning System
Main Flask application entry point with Blueprint registration.
"""

from flask import Flask, render_template
from flask_cors import CORS
from routes.generate import generate_bp
from routes.classify import classify_bp
from routes.clusters import clusters_bp


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    CORS(app)

    # Register API blueprints with /api prefix
    app.register_blueprint(generate_bp, url_prefix='/api')
    app.register_blueprint(classify_bp, url_prefix='/api')
    app.register_blueprint(clusters_bp, url_prefix='/api')

    # Serve the frontend dashboard
    @app.route('/')
    def index():
        return render_template('index.html')

    return app


if __name__ == '__main__':
    app = create_app()
    print("\n🧠 NeuralCanvas is running!")
    print("📍 Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
