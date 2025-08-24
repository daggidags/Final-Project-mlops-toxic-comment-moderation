import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
import mlflow
import mlflow.sklearn

# Set base path
base_dir = Path(__file__).resolve().parent.parent
data_path = base_dir / "data" / "train.csv"
model_path = base_dir / "api" / "toxicity_model.pkl"

# Load dataset
df = pd.read_csv(data_path)
X = df["comment_text"]
y = df["toxic"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# Define pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('clf', LogisticRegression(max_iter=1000))
])

# Start MLflow experiment
mlflow.set_experiment("toxicity-moderation")
with mlflow.start_run():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Log params and metrics
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

    # Log model in MLflow
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        registered_model_name="ToxicCommentModel"
    )

    # Save model for FastAPI
    joblib.dump(pipeline, model_path)

    mlflow.log_artifact(model_path)