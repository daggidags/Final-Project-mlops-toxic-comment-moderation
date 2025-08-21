# src/train.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn

 # may need to adjust to get to correct working directory!
base_dir = Path(__file__).resolve().parent.parent
data_path = base_dir / "data" / "train.csv"

df = pd.read_csv(data_path)

X = df["comment_text"]
y = df["toxic"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# logistic model?
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('clf', LogisticRegression(max_iter=1000))
])

mlflow.set_experiment("toxicity-moderation")
with mlflow.start_run():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mlflow.log_params({
        "vectorizer": "tfidf",
        "classifier": "logreg",
        "max_features": 10000,
        "max_iter": 1000
    })
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    })
    mlflow.sklearn.log_model(pipeline, artifact_path="model", registered_model_name="ToxicCommentModel")