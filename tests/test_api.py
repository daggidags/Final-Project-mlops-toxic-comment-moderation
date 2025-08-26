from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_toxic():
    response = client.post(
        "/predict",
        json={
            "text": "Don't look, come or think of coming back! Tosser.",
            "true_label": 1  # toxic
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_label" in data
    assert isinstance(data["predicted_label"], int)


def test_non_toxic():
    response = client.post(
        "/predict",
        json={
            "text": "That is correct.",
            "true_label": 0  # non-toxic
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_label" in data
    assert isinstance(data["predicted_label"], int)
