"""
Integration tests for FastAPI serving layer (health probes, schemas, endpoints).
"""

import pytest
from fastapi.testclient import TestClient
from lunar_lander.api import app, app_state
from lunar_lander.policy import PolicyEngine


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoints(client):
    r_healthz = client.get("/healthz")
    assert r_healthz.status_code == 200
    assert r_healthz.json() == {"status": "HEALTHY"}

    r_livez = client.get("/livez")
    assert r_livez.status_code == 200
    assert r_livez.json() == {"status": "LIVE"}

    r_readyz = client.get("/readyz")
    assert r_readyz.status_code == 200
    assert r_readyz.json() == {"status": "READY"}


def test_metrics_endpoint(client):
    r_metrics = client.get("/metrics")
    assert r_metrics.status_code == 200
    data = r_metrics.json()
    assert data["status"] == "OPERATIONAL"
    assert "uptime_seconds" in data
    assert "total_requests" in data
    assert "avg_latency_ms" in data


def test_predict_nominal_telemetry(client):
    payload = {
        "x": 0.05,
        "y": 0.75,
        "vx": -0.02,
        "vy": -0.15,
        "angle": 0.01,
        "angular_velocity": 0.0,
        "leg1_contact": 0.0,
        "leg2_contact": 0.0,
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "action" in data
    assert data["action"] in [0, 1, 2, 3]
    assert "action_name" in data
    assert "confidence" in data
    assert "critic_value" in data
    assert "safety" in data
    assert data["safety"]["overridden"] is False
    assert data["latency_ms"] >= 0.0


def test_predict_safety_intervention(client):
    # State drifting left near ground: neural policy wants horizontal correction (action 3),
    # but critical vertical descent velocity forces touchdown brake override (action 2)
    payload = {
        "x": -0.5,
        "y": 0.05,
        "vx": -1.0,
        "vy": -1.3,
        "angle": 0.0,
        "angular_velocity": 0.0,
        "leg1_contact": 0.0,
        "leg2_contact": 0.0,
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["safety"]["overridden"] is True
    assert data["action"] == 2  # Overridden to main engine braking
    assert "TOUCHDOWN_DESCENT_BRAKE" in data["safety"]["reason"]


def test_predict_validation_error(client):
    # Malformed payload: x exceeds physical validation bound (-5 to +5)
    invalid_payload = {
        "x": 999.0,
        "y": 0.5,
        "vx": 0.0,
        "vy": 0.0,
        "angle": 0.0,
        "angular_velocity": 0.0,
        "leg1_contact": 0.0,
        "leg2_contact": 0.0,
    }
    response = client.post("/v1/predict", json=invalid_payload)
    assert response.status_code == 422  # Pydantic validation error


def test_predict_vector_endpoint(client):
    payload = {"state": [0.0, 0.5, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0]}
    response = client.post("/v1/predict_vector", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "action" in data


def test_batch_predict_endpoint(client):
    telemetry = {
        "x": 0.0,
        "y": 0.6,
        "vx": 0.0,
        "vy": -0.1,
        "angle": 0.0,
        "angular_velocity": 0.0,
        "leg1_contact": 0.0,
        "leg2_contact": 0.0,
    }
    batch_payload = {"telemetries": [telemetry] * 5}
    response = client.post("/v1/batch_predict", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["batch_size"] == 5
    assert len(data["predictions"]) == 5
    assert data["total_latency_ms"] > 0.0
