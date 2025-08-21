# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

model_name = "ToxicCommentModel"
model_stage = "None"  # or "Staging" or "Production"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")

class CommentInput(BaseModel):
    comment_text: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(input_data: CommentInput):
    input_df = pd.DataFrame({"comment_text": [input_data.comment_text]})
    prediction = model.predict(input_df)[0]
    return {"toxic": bool(prediction)}
