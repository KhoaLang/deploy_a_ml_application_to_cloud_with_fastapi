import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture(scope="session")
def client():
    """
    Create a TestClient instance
    """
    client = TestClient(app)
    return client


def test_get_method(client):
    """
    Test get method
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "A machine learning deployment project!"}


def test_prediction_for_above_50k(client):
    """
    Test for prediction result is salary above 50K
    """
    body = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 5000,
        "capital-loss": 100,
        "hours-per-week": 50,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=body)

    assert response.status_code == 200
    assert response.json() == {'Output': '>50K'}


def test_prediction_for_below_50k(client):
    """
    Test for prediction result is salary below 50K
    """
    body = {
        "age": 23,
        "workclass": "Private",
        "fnlgt": 122272,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 30,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=body)

    assert response.status_code == 200
    assert response.json() == {'Output': '<=50K'}
