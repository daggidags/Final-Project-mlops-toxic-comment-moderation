import pytest
from fastapi.testclient import TestClient
from src import app

client = TestClient(app)

#Unit Tests - preprocessing data


#Interactive Test- verify endpoints
#Toxic Comment
def test_toxic():
    #testing a toxic comment #double check input type- json?
    response = client.post("/predict", json={
        comment_text = "Don't look, come or think of coming back! Tosser."}

    assert response.status_code == 200
    assert prediction[0] == 1

#Nontoxic comment
def test_non_toxic():
    #testing a toxic comment #double check input type- json?
    response = client.post("/predict", json={
        comment_text = "That is correct."}

    assert response.status_code == 200
    assert prediction[0] == 0