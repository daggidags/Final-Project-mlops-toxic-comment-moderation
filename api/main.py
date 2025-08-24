from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from pathlib import Path
import os
import json
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Toxic Comment Classification API",
    description="Classifies user comments as toxic or non-toxic and logs results for monitoring",
    version="1.0"
)

# Load trained model
model_path = Path(__file__).resolve().parent / "sentiment_model.pkl"
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = joblib.load(model_path)

# Define input format
class PredictionInput(BaseModel):
    text: str
    true_label: str  # optional, for logging and monitoring

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Predict toxicity endpoint
@app.post("/predict")
def predict_toxicity(input: PredictionInput):
    try:
        prediction = model.predict([input.text])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Create logs directory if it doesn’t exist
    os.makedirs("/logs", exist_ok=True)

    # Log prediction data for monitoring
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_text": input.text,
        "true_label": input.true_label,
        "predicted_label": int(prediction)
    }

    with open("/logs/prediction_logs.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {"predicted_label": int(prediction)}
