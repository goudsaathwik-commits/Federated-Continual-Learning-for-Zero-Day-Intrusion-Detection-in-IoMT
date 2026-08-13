import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_web_app_index(client):
    """Verify web dashboard index page renders successfully."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"IoMT Zero-Day IDS" in response.data
    assert b"Federated Learning" in response.data

def test_web_app_metrics_api(client):
    """Verify metrics API returns valid JSON payload."""
    response = client.get('/api/metrics')
    assert response.status_code == 200
    data = response.get_json()
    assert "federated" in data or "centralized" in data

def test_web_app_predict_anomaly_api(client):
    """Verify live zero-day anomaly simulator endpoint."""
    response = client.post('/api/predict_anomaly', json={"type": "zero_day"})
    assert response.status_code == 200
    data = response.get_json()
    assert "energy_score" in data
    assert "is_anomaly" in data
    assert data["is_anomaly"] is True
