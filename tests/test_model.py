import joblib
import os
import pytest

MODEL_PATH = os.path.join("api", "toxicity_model.pkl")

def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), "toxicity_model.pkl is missing!"

def test_model_can_predict():
    model = joblib.load(MODEL_PATH)
    sample_text = ["You are so stupid!"]
    prediction = model.predict(sample_text)
    assert prediction[0] in [0, 1], "Model returned unexpected value"
