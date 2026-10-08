import streamlit as st
import requests

st.title("Text Classification App")
text_input = st.text_area("Enter text to classify:")

if st.button("Predict"):
    if text_input.strip():
        try:
            # Flask API Endpoint
            url = "http://127.0.0.1:5000/predict"
            response = requests.post(url, json={"text": text_input})

            if response.status_code == 200:
                data = response.json()
                st.success(f"Prediction: **{data['prediction']}**")
                if data.get("confidence"):
                    st.json(data["confidence"])
            else:
                st.error(f"API Error ({response.status_code}): {response.json().get('error')}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to Flask API. Ensure 'python app.py' is running on port 5000.")
    else:
        st.warning("Please enter text before submitting.")
