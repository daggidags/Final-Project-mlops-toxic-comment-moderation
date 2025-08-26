import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

# Path to the shared logs volume
LOGS_PATH = "/logs/prediction_logs.json"

st.set_page_config(page_title="Toxicity Model Monitor", layout="wide")
st.title("Toxicity Model Monitoring Dashboard")
st.markdown("This dashboard visualizes predictions and tracks model performance in real-time.")

# Check if logs exist
if not os.path.exists(LOGS_PATH):
    st.warning("No logs found yet. Make some predictions using the API!")
else:
    # Load prediction logs
    logs = []
    with open(LOGS_PATH, "r") as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except:
                pass

    if len(logs) == 0:
        st.info("No prediction data available yet.")
    else:
        df = pd.DataFrame(logs)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        st.sidebar.header("Filters")

        if "predicted_sentiment" in df.columns:
            sentiment_filter = st.sidebar.multiselect(
                "Filter by Predicted Sentiment:",
                options=df["predicted_sentiment"].dropna().unique(),
                default=list(df["predicted_sentiment"].dropna().unique())
            )

            df_filtered = df[df["predicted_sentiment"].isin(sentiment_filter)]

            # Show latest predictions
            st.subheader("Recent Predictions")
            st.dataframe(df_filtered.sort_values("timestamp", ascending=False).tail(10))

            # Plot prediction counts
            st.subheader("Predictions Over Time")
            chart_df = (
                df_filtered
                .groupby([df_filtered["timestamp"].dt.date, "predicted_sentiment"])
                .size()
                .reset_index(name="count")
                .rename(columns={"timestamp": "date"})
            )
            st.line_chart(
                chart_df.pivot(index="date", columns="predicted_sentiment", values="count").fillna(0)
            )

            # Compare predicted vs true sentiment
            if "true_sentiment" in df_filtered.columns and df_filtered["true_sentiment"].notna().any():
                st.subheader("Prediction Accuracy")
                accuracy = (df_filtered["predicted_sentiment"] == df_filtered["true_sentiment"]).mean() * 100
                st.metric("Model Accuracy", f"{accuracy:.2f}%")

                confusion = pd.crosstab(df_filtered["true_sentiment"], df_filtered["predicted_sentiment"])
                st.write("Confusion Matrix")
                st.dataframe(confusion)
        else:
            st.warning("No 'predicted_sentiment' data found in logs yet. Make some predictions via the API or frontend.")

st.markdown("---")
st.caption("Toxicity Model Monitoring Dashboard — Powered by Streamlit")
