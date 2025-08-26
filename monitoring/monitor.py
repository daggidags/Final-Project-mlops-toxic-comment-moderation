import streamlit as st
import pandas as pd
import psycopg2
import os
import plotly.express as px

# --- DB CONNECTION ---
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        port=os.getenv("DB_PORT", 5432),
    )

# --- LOAD DATA ---
@st.cache_data(ttl=60)
def load_prediction_logs():
    conn = get_db_connection()
    query = """
        SELECT timestamp, request_text, predicted_label, true_label
        FROM prediction_logs
        ORDER BY timestamp DESC
        LIMIT 500;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# --- STREAMLIT DASHBOARD ---
st.set_page_config(page_title="Toxicity Model Monitoring", layout="wide")
st.title("🧠 Toxicity Model Monitoring Dashboard")

try:
    df = load_prediction_logs()

    if df.empty:
        st.warning("No prediction logs found yet. Make some /predict requests!")
    else:
        # Show latest logs
        st.subheader("📄 Latest Prediction Logs")
        st.dataframe(df)

        # Prediction counts per label
        st.subheader("📊 Predictions by Label")
        counts = df["predicted_label"].value_counts().reset_index()
        counts.columns = ["Predicted Label", "Count"]
        fig1 = px.bar(counts, x="Predicted Label", y="Count", color="Predicted Label", title="Distribution of Predictions")
        st.plotly_chart(fig1, use_container_width=True)

        # Accuracy over time
        st.subheader("📈 Prediction Accuracy Over Time")
        df["correct"] = (df["predicted_label"] == df["true_label"]).astype(int)
        accuracy_df = df.groupby(df["timestamp"].dt.date)["correct"].mean().reset_index()
        accuracy_df.columns = ["Date", "Accuracy"]
        fig2 = px.line(accuracy_df, x="Date", y="Accuracy", markers=True, title="Model Accuracy Over Time")
        st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.error(f"❌ Error loading data: {e}")
