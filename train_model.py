# train_model.py

import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import os

# Define the path for the model
MODEL_PATH = 'house_price_model.pkl'

def train_and_save_model():
    """
    Trains a model on the California Housing dataset and saves it.
    """
    print("Loading dataset...")
    # Load the California Housing dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    print(f"Dataset has {X.shape[0]} samples and {X.shape[1]} features.")
    print("Feature names:", housing.feature_names)

    # Initialize and train the model
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X, y)
    print("Model training complete.")

    # Save the model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    # We only train if the model doesn't already exist
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()
    else:
        print(f"Model file '{MODEL_PATH}' already exists. Skipping training.")
        # Optional: You could still print feature names for reference
        housing = fetch_california_housing()
        print("Feature names for frontend:", housing.feature_names)