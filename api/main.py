from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import psycopg2
from datetime import datetime
import os

app = FastAPI()

# Load sentiment model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "toxicity_model.pkl")
model = joblib.load(MODEL_PATH)

# Define input data model
class PredictionInput(BaseModel):
    text: str
    true_label: int

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "toxicity-db.cdowqssegxo6.us-east-1.rds.amazonaws.com"),
        port=os.getenv("DB_PORT", "5432"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "finalproject"),
        dbname=os.getenv("DB_NAME", "postgres")
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict_sentiment(input: PredictionInput):
    try:
        prediction = model.predict([input.text])[0]
        prediction = int(prediction)  

        # Log prediction to RDS
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO prediction_logs (timestamp, request_text, predicted_label, true_label)
            VALUES (%s, %s, %s, %s);
        """, (datetime.utcnow(), input.text, int(prediction), input.true_label))

        conn.commit()
        cur.close()
        conn.close()

        return {"text": input.text, "predicted_label": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
