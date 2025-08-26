import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

# Streamlit app
st.set_page_config(page_title="Toxic Comment Moderation", page_icon="🧠", layout="centered")

st.title("Toxic Comment Moderation Dashboard")
st.write("Enter a comment below to classify whether it is toxic or safe.")

# Input field
user_input = st.text_area("Enter your comment:", height=150)

# Submit button
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a comment first.")
    else:
        # Send request to FastAPI
        try:
            payload = {"text": user_input, "true_label": None}  # No label yet for user predictions
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                prediction = response.json()
                st.success(f"**Prediction:** {prediction}")

            else:
                st.error(f"Error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Unable to connect to FastAPI backend. Is it running?")
