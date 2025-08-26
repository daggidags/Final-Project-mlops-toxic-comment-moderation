from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime
from typing import Optional

app = FastAPI()

# Load sentiment model
model = joblib.load("toxicity_model.pkl")

# Database connection settings
DB_HOST = os.getenv("DB_HOST", "toxicity-db.cdowqssegxo6.us-east-1.rds.amazonaws.com")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "finalproject")
DB_PORT = 5432

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT,
        cursor_factory=RealDictCursor
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Prediction input model
class PredictionInput(BaseModel):
    text: str
    true_label: Optional[int] = None

@app.post("/predict")
def predict_toxicity(input: PredictionInput):
    try:
        # Make prediction and convert to native Python int
        prediction = int(model.predict([input.text])[0])

        # Create database connection
        conn = get_db_connection()
        cursor = conn.cursor()

        # Insert log entry (convert true_label too)
        cursor.execute("""
            INSERT INTO prediction_logs (timestamp, request_text, predicted_label, true_label)
            VALUES (%s, %s, %s, %s)
        """, (
            datetime.utcnow(),
            input.text,
            prediction,
            int(input.true_label) if input.true_label is not None else None
        ))

        conn.commit()
        cursor.close()
        conn.close()

        # Return JSON-safe response
        return {"predicted_label": prediction}

    except Exception as e:
        print(f"Failed to log prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
