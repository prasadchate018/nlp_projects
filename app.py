import streamlit as st
import requests

# Page setup
st.set_page_config(
    page_title="AI Sentiment & Text Classifier",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Glassmorphism Background, Card Glow, and Modern Typography
st.markdown("""
<style>
    /* Gradient Background Effect */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
        color: #f8fafc;
    }

    /* Glassmorphism Outer Card */
    div.stCard {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Headers with Gradient Text */
    .gradient-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .sub-text {
        color: #94a3b8;
        text-align: center;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* Primary Prediction Button Glow */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: #ffffff;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        border: none;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.4);
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(168, 85, 247, 0.6);
        background: linear-gradient(90deg, #4f46e5 0%, #9333ea 100%);
    }

    /* Custom Metric Display */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Main Container UI
st.markdown("<h1 class='gradient-header'>Multinomial Naive Bayes Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Powered by Flask API & Scikit-Learn Inference Engine</p>", unsafe_allow_html=True)

# Flask Server Config
FLASK_API_URL = "http://127.0.0.1:5000/predict"

with st.container():
    st.markdown("<div class='stCard'>", unsafe_allow_html=True)
    
    st.subheader("📊 Input Features Matrix")
    st.caption("Provide numerical feature attributes for model inference:")
    
    # Dynamic inputs for model features (480 parameters supported)
    col1, col2 = st.columns(2)
    with col1:
        f1 = st.number_input("Feature Vector [0]", value=0.15, step=0.01)
        f2 = st.number_input("Feature Vector [1]", value=1.20, step=0.01)
    with col2:
        f3 = st.number_input("Feature Vector [2]", value=0.00, step=0.01)
        f4 = st.number_input("Feature Vector [3]", value=0.85, step=0.01)
    
    # Padding array to match the model requirement (480 features)
    features_list = [f1, f2, f3, f4] + [0.0] * 476

    st.write("")
    
    if st.button("🚀 Run Classification"):
        with st.spinner("Communicating with Flask Backend..."):
            try:
                response = requests.post(FLASK_API_URL, json={"features": features_list}, timeout=5)
                
                if response.status_code == 200:
                    result = response.json()
                    pred_class = result.get("prediction")
                    probs = result.get("probabilities", {})

                    st.markdown("---")
                    st.success("✅ Classification Successful!")
                    
                    # Display Output
                    st.markdown(f"### Predicted Class: **`{pred_class.upper()}`**")
                    
                    # Probability Distribution Visualizer
                    st.write("#### Confidence Metrics")
                    for cls_name, prob_val in probs.items():
                        st.write(f"**{cls_name.title()}**: {prob_val * 100:.2f}%")
                        st.progress(float(prob_val))
                else:
                    st.error(f"Error {response.status_code}: {response.json().get('error', 'Unknown Error')}")
            
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Could not connect to Flask API. Ensure `python app.py` is running on port 5000.")

    st.markdown("</div>", unsafe_allow_html=True)
