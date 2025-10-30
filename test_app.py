# test_app.py

import pytest
from app import app  # Import the 'app' instance from your app.py file
import json

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    with app.test_client() as client:
        yield client

def test_home_route(client):
    """Test the home page."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"House Price Prediction" in response.data

def test_predict_route_success(client):
    """Test a successful prediction."""
    # These are the same default features from your index.html
    test_data = {
        "features": [8.3, 41, 6.9, 1.0, 322, 2.5, 37.88, -122.23]
    }
    response = client.post('/predict', json=test_data)
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert isinstance(data['prediction'], float)

def test_predict_route_bad_data(client):
    """Test a prediction with missing/bad data."""
    test_data = {"features": [1, 2, 3]} # Not enough features
    
    response = client.post('/predict', json=test_data)
    
    # Our app.py returns 400 on an exception
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data