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

# Start MLflow experiment
mlflow.set_experiment("toxicity-moderation")

# First model pipeline
pipeline1 = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('clf', LogisticRegression(penalty=None, max_iter=1000))
])

# Second model pipeline
pipeline2 = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('clf', LogisticRegression(penalty='l2', max_iter=1000))
])

# Start main MLflow run
with mlflow.start_run(run_name="Baseline vs L2 Logistic Regression"):

    # first model
    pipeline1.fit(X_train, y_train)
    y_pred1 = pipeline1.predict(X_test)

    mlflow.log_params({
        "model": "pipeline1",
        "vectorizer": "tfidf",
        "classifier": "logreg",
        "penalty": None,
        "max_features": 10000,
        "max_iter": 1000
    })
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, y_pred1),
        "f1": f1_score(y_test, y_pred1)
    })

    mlflow.sklearn.log_model(
        pipeline1,
        artifact_path="model_pipeline1",
        registered_model_name="ToxicCommentModel_v1"
    )

    # second model
    with mlflow.start_run(run_name="Pipeline2-L2-Regularization", nested=True):
        pipeline2.fit(X_train, y_train)
        y_pred2 = pipeline2.predict(X_test)

        mlflow.log_params({
            "model": "pipeline2",
            "vectorizer": "tfidf",
            "classifier": "logreg",
            "penalty": "l2",
            "max_features": 10000,
            "max_iter": 1000
        })
        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, y_pred2),
            "f1": f1_score(y_test, y_pred2)
        })

        mlflow.sklearn.log_model(
            pipeline2,
            artifact_path="model_pipeline2",
            registered_model_name="ToxicCommentModel_v2"
        )

    # save model
    joblib.dump(pipeline1, model_path)
    mlflow.log_artifact(model_path)
