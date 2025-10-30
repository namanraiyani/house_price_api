# app.py

import joblib
from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import logging
import logging.config

# --- 1. Setup Metrics (for Prometheus/Grafana) ---
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Histogram

# --- 2. Setup Logging ---
try:
    logging.config.fileConfig('logging.ini')
    logger = logging.getLogger(__name__)
    logger.info("Logging configured from file.")
except Exception as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not load logging.ini ({e}). Using basic config.")


app = Flask(__name__)

# --- 3. Initialize Metrics ---
metrics = PrometheusMetrics(app) # This handles all basic /metrics
# Define custom buckets for house prices (e.g., $0 to $500k in $50k steps)
# The library expects the raw model output (e.g., 4.5 for $450k)
buckets = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, float("inf"))
PREDICTION_VALUE_HISTOGRAM = Histogram(
    'house_price_prediction_value',
    'Histogram of predicted house price values',
    buckets=buckets
)
logger.info("Prometheus metrics endpoint initialized at /metrics")

# --- 4. Model Loading ---
MODEL_PATH = 'house_price_model.pkl'
model = None

if os.path.exists(MODEL_PATH):
    logger.info(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully.")
else:
    logger.error(f"Error: Model file '{MODEL_PATH}' not found.")
    logger.error("Please run train_model.py first to create the model.")


# --- 5. API Routes ---

@app.route('/')
def home():
    """Serves the frontend HTML page."""
    logger.info(f"GET / - Serving homepage (index.html)")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests from the frontend."""
    logger.info("POST /predict - Received prediction request.")
    
    if model is None:
        logger.error("Prediction failed: Model is not loaded.")
        return jsonify({'error': 'Model is not loaded. Run train_model.py.'}), 500

    try:
        data = request.get_json()
        features_list = data['features']
        logger.debug(f"Received features: {features_list}") 
        
        final_features = np.array(features_list).astype(float).reshape(1, -1)
        prediction = model.predict(final_features)
        output = prediction[0] # e.g., 4.52

        # --- 6. RECORD THE METRIC ---
        PREDICTION_VALUE_HISTOGRAM.observe(output)
        
        logger.info(f"Prediction successful. Output: {output}")
        return jsonify({'prediction': output})

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 400

# --- Run the App ---
if __name__ == "__main__":
    # Note: When running with gunicorn (in Docker), this part is not executed.
    logger.info("Starting Flask development server...")
    app.run(debug=True, port=5000) # Gunicorn will use its own port