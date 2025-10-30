# app.py

import joblib
from flask import Flask, request, jsonify, render_template
import numpy as np
import os

app = Flask(__name__)

# --- Model Loading ---
MODEL_PATH = 'house_price_model.pkl'
model = None

# Load the model only if it exists
if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
else:
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    print("Please run train_model.py first to create the model.")
    # You might want to exit or handle this error more gracefully
    # For this demo, we'll let it run, but /predict will fail.

# --- API Routes ---

@app.route('/')
def home():
    """Serves the frontend HTML page."""
    # Renders the HTML file from the 'templates' folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests from the frontend."""
    
    if model is None:
        # Return a 500 Internal Server Error if the model isn't loaded
        return jsonify({'error': 'Model is not loaded. Run train_model.py.'}), 500

    try:
        # Get the JSON data sent from the frontend
        data = request.get_json()
        
        # Extract the 'features' list from the JSON payload
        features_list = data['features']
        
        # Convert list of string/number inputs to a numpy array of floats
        # Sklearn models expect a 2D array, so we reshape [1, -1]
        final_features = np.array(features_list).astype(float).reshape(1, -1)

        # Make prediction
        prediction = model.predict(final_features)

        # Get the single prediction value
        output = prediction[0]

        # Return the prediction as JSON
        # The Flask server automatically sets the status code to 200 (OK)
        return jsonify({'prediction': output})

    except Exception as e:
        # Handle potential errors (e.g., bad data format)
        print(f"Error during prediction: {e}")
        # Return a 400 Bad Request error
        return jsonify({'error': str(e)}), 400

# --- Run the App ---

if __name__ == "__main__":
    # Setting debug=True gives helpful error messages
    app.run(debug=True)