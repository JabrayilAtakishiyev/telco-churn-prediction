import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.config import MODEL_PATH

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="Trained model file is not available.")


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert 0.0 < body["threshold"] < 1.0


def test_predict_returns_valid_probability(client, sample_customer):
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 200
    body = response.json()
    assert body["customerID"] == "TEST-0001"
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["predicted_churn"] in (0, 1)


def test_high_risk_customer_scores_above_low_risk_customer(client, sample_customer):
    high_risk = dict(
        sample_customer,
        tenure=1,
        InternetService="Fiber optic",
        Contract="Month-to-month",
        PaymentMethod="Electronic check",
        MonthlyCharges=95.0,
        TotalCharges=95.0,
    )
    low_risk = dict(
        sample_customer,
        tenure=60,
        InternetService="DSL",
        Contract="Two year",
        PaymentMethod="Credit card (automatic)",
        MonthlyCharges=55.0,
        TotalCharges=3300.0,
    )
    high_probability = client.post("/predict", json=high_risk).json()["churn_probability"]
    low_probability = client.post("/predict", json=low_risk).json()["churn_probability"]
    assert high_probability > low_probability


def test_predict_without_total_charges(client, sample_customer):
    del sample_customer["TotalCharges"]
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 200


def test_invalid_category_returns_422(client, sample_customer):
    sample_customer["Contract"] = "Weekly"
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 422


def test_negative_tenure_returns_422(client, sample_customer):
    sample_customer["tenure"] = -1
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 422


def test_batch_prediction(client, sample_customer):
    second = dict(sample_customer, customerID="TEST-0002", tenure=40)
    response = client.post("/predict/batch", json={"customers": [sample_customer, second]})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert [item["customerID"] for item in predictions] == ["TEST-0001", "TEST-0002"]


def test_empty_batch_returns_422(client):
    response = client.post("/predict/batch", json={"customers": []})
    assert response.status_code == 422
