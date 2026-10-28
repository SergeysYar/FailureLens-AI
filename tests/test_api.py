from failurelens.api import app
from fastapi.testclient import TestClient


def test_health_and_summary_contract():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    response = client.get("/api/v1/summary")
    assert response.status_code == 200
    assert {"devices", "observations", "demo"} <= response.json().keys()


def test_predict_rejects_invalid_horizon():
    client = TestClient(app)
    response = client.post("/api/v1/predict", json={"date": "2024-01-01", "smart_5": 1, "smart_187": 1,
                                                      "temperature": 30, "read_errors": 0, "horizon_days": 14})
    assert response.status_code == 422
