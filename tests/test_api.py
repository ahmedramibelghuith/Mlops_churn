from types import SimpleNamespace
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend import main as api


@pytest.fixture
def client(monkeypatch):
    class Model:
        metadata = SimpleNamespace(metadata={"input_format": "raw-v1"})

        def predict(self, frame):
            assert "Churn" not in frame
            return np.zeros(len(frame), dtype=int)

    monkeypatch.setenv("MODEL_URI", "runs:/test/ML_models")
    monkeypatch.setattr(api, "configure_tracking", lambda: None)
    monkeypatch.setattr(api.mlflow.pyfunc, "load_model", lambda uri: Model())
    with TestClient(api.app) as test_client:
        yield test_client


def test_json_and_csv_agree(client):
    data = {"Age": 22, "Gender": "Female", "Tenure": 25,
            "Usage Frequency": 14, "Support Calls": 4, "Payment Delay": 27,
            "Subscription Type": "Basic", "Contract Length": "Monthly",
            "Total Spend": 598.5, "Last Interaction": 9}
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    csv = ",".join(data) + "\n" + ",".join(map(str, data.values())) + "\n"
    batch = client.post("/predict/csv", files={"file": ("customers.csv", csv, "text/csv")})
    assert batch.status_code == 200
    assert batch.json() == response.json() == {"predictions": [0]}


def test_invalid_category(client):
    assert client.post("/predict", json={"Gender": "invalid"}).status_code == 422


def test_invalid_csv(client):
    response = client.post("/predict/csv", files={"file": ("bad.csv", "Age\n22\n", "text/csv")})
    assert response.status_code == 422


def test_unavailable_model_returns_503(client, monkeypatch):
    monkeypatch.setattr(api, "model", None)
    assert client.get("/health").status_code == 503
