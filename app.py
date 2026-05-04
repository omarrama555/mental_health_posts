import streamlit as st
import pickle
import numpy as np
import time
from datetime import datetime

# =============================
# Page Config
# =============================
st.set_page_config(
    page_title="Mental Health Crisis Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# Custom CSS (Animation + UI)
# =============================
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: 700;
    color: #4CAF50;
    animation: fadeIn 1.5s ease-in-out;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #111;
    box-shadow: 0 0 15px rgba(0,0,0,0.2);
    margin-bottom: 20px;
    animation: slideUp 0.6s ease-in-out;
}

@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

@keyframes slideUp {
    from {transform: translateY(20px); opacity:0;}
    to {transform: translateY(0); opacity:1;}
}
</style>
""", unsafe_allow_html=True)

# =============================
# Load Models (Cached)
# =============================
@st.cache_resource
def load_models():
    with open("saved_models/sbert_lr.pkl", "rb") as f:
        sbert_model = pickle.load(f)

    with open("saved_models/lr_augmented.pkl", "rb") as f:
        lr_model = pickle.load(f)

    with open("saved_models/tfidf_augmented.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open("saved_models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return sbert_model, lr_model, tfidf, label_encoder

sbert_model, lr_model, tfidf, label_encoder = load_models()

# =============================
# Sidebar
# =============================
st.sidebar.title("Settings")
mode = st.sidebar.selectbox("Choose Model Mode", ["TF-IDF + LR", "SBERT + LR"])

show_prob = st.sidebar.checkbox("Show Probabilities", True)
show_history = st.sidebar.checkbox("Show History", True)

# =============================
# Session State
# =============================
if "history" not in st.session_state:
    st.session_state.history = []

# =============================
# Title
# =============================
st.markdown('<div class="main-title">Mental Health Crisis Detection</div>', unsafe_allow_html=True)

# =============================
# Input Section
# =============================
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    user_input = st.text_area(
        "Enter text",
        placeholder="Type a message...",
        height=150
    )

    col1, col2 = st.columns(2)

    predict_btn = col1.button("Analyze")
    clear_btn = col2.button("Clear")

    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# Clear
# =============================
if clear_btn:
    st.session_state.history = []
    st.rerun()

# =============================
# Prediction Function
# =============================
def predict(text):
    if mode == "TF-IDF + LR":
        vec = tfidf.transform([text])
        pred = lr_model.predict(vec)[0]
        prob = lr_model.predict_proba(vec)[0]
    else:
        emb = sbert_model.encode([text])
        pred = lr_model.predict(emb)[0]
        prob = lr_model.predict_proba(emb)[0]

    label = label_encoder.inverse_transform([pred])[0]
    return label, prob

# =============================
# Real-Time Features (10)
# =============================
if predict_btn and user_input.strip() != "":

    # 1) Loading animation
    with st.spinner("Analyzing..."):
        time.sleep(1)

    label, prob = predict(user_input)

    # 2) Result Card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction Result")
    st.success(f"Label: {label}")
    st.markdown('</div>', unsafe_allow_html=True)

    # 3) Probability display
    if show_prob:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Confidence Scores")
        for i, p in enumerate(prob):
            st.progress(float(p))
        st.markdown('</div>', unsafe_allow_html=True)

    # 4) Word count
    word_count = len(user_input.split())
    st.write(f"Word Count: {word_count}")

    # 5) Character count
    st.write(f"Character Count: {len(user_input)}")

    # 6) Risk flag logic
    if "kill" in user_input.lower() or "die" in user_input.lower():
        st.error("High Risk Keywords Detected")

    # 7) Timestamp logging
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 8) Save history
    st.session_state.history.append({
        "text": user_input,
        "label": label,
        "time": timestamp
    })

# =============================
# History Panel
# =============================
if show_history and len(st.session_state.history) > 0:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction History")

    for item in reversed(st.session_state.history[-5:]):
        st.write(f"[{item['time']}] {item['text']} → {item['label']}")

    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# Footer
# =============================
st.markdown("---")
st.caption("Deployed using Streamlit Cloud | Production Ready App")
