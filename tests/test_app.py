import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

os.environ["IA_API_KEY"] = "test-key"

from app import app  # noqa: E402


REQUEST = json.loads((Path(__file__).parents[1] / "example_request.json").read_text())


def test_health_loads_model():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_requires_api_key():
    with TestClient(app) as client:
        response = client.post("/v1/predict", json=REQUEST)
    assert response.status_code == 401


def test_predict_returns_model_output():
    with TestClient(app) as client:
        response = client.post("/v1/predict", headers={"X-API-Key": "test-key"}, json=REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"]
    assert 0 <= body["rain_probability"] <= 1
    assert body["horizon_hours"] == 1
