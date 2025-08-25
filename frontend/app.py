import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# ----------------------------
# CONFIGURATION
# ----------------------------
FASTAPI_URL = "http://localhost:8000/predict"  # Update later when deployed

# ----------------------------
# PAGE SETUP
# ----------------------------
st.set_page_config(
    page_title="Toxicity Prediction App",
    page_icon="🤖",
    layout="centered"
)

st.title("Toxic Comment Classifier")
st.write("Enter a comment below and see if it's classified as **toxic** or **non-toxic**.")

user_input = st.text_area(
    "Enter your comment:",
    placeholder="Type something here..."
)

if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# PREDICTION BUTTON
# ----------------------------
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
        try:
            # Send POST request to FastAPI
            response = requests.post(
                FASTAPI_URL,
                json={"text": user_input, "true_label": "unknown"}
            )

            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction", "Unknown")

                # Display result
                st.success(f"**Prediction:** {prediction}")

                # Store prediction in history
                st.session_state.history.insert(0, {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "text": user_input,
                    "prediction": prediction
                })

            else:
                st.error(f"Error: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to FastAPI backend. Make sure it's running.")

# ----------------------------
# PREDICTION HISTORY
# ----------------------------
if st.session_state.history:
    st.subheader("Recent Predictions")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)
