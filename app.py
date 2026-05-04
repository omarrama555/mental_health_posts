# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import os
import plotly.graph_objects as go
import time

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="MindGuard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 3D ANIMATIONS & PROFESSIONAL CSS
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* 3D Animation Keyframes */
    @keyframes fadeInUp {
        0% {
            opacity: 0;
            transform: translateY(30px) scale(0.95);
        }
        100% {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(59,130,246,0.4);
        }
        70% {
            box-shadow: 0 0 0 15px rgba(59,130,246,0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(59,130,246,0);
        }
    }
    
    @keyframes float {
        0% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-10px);
        }
        100% {
            transform: translateY(0px);
        }
    }
    
    @keyframes glowPulse {
        0% {
            opacity: 0.3;
            filter: blur(20px);
        }
        100% {
            opacity: 0.8;
            filter: blur(30px);
        }
    }
    
    /* Main container animation */
    .animated-container {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Header with 3D effect */
    .header-3d {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        padding: 2rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 35px -10px rgba(0,0,0,0.2);
        border: 1px solid rgba(59,130,246,0.2);
        position: relative;
        overflow: hidden;
    }
    
    .header-3d::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(59,130,246,0.1) 0%, transparent 70%);
        animation: glowPulse 3s ease-in-out infinite alternate;
    }
    
    .header-3d h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 600;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }
    
    .header-3d p {
        color: #94a3b8;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        position: relative;
        z-index: 1;
    }
    
    /* 3D Cards */
    .card-3d {
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
        animation: fadeInUp 0.5s ease-out;
    }
    
    .card-3d:hover {
        transform: translateY(-5px);
        box-shadow: 0 30px 60px -12px rgba(0,0,0,0.2);
    }
    
    /* Result Cards with 3D depth */
    .result-crisis {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border-radius: 24px;
        padding: 2rem;
        color: white;
        animation: fadeInUp 0.5s ease-out, pulse 2s infinite;
        box-shadow: 0 20px 40px -10px rgba(127,29,29,0.4);
        border: 1px solid rgba(248,113,113,0.3);
    }
    
    .result-support {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        border-radius: 24px;
        padding: 2rem;
        color: white;
        animation: fadeInUp 0.5s ease-out;
        box-shadow: 0 20px 40px -10px rgba(30,58,138,0.4);
        border: 1px solid rgba(96,165,250,0.3);
    }
    
    .result-neutral {
        background: linear-gradient(135deg, #064e3b 0%, #047857 100%);
        border-radius: 24px;
        padding: 2rem;
        color: white;
        animation: fadeInUp 0.5s ease-out;
        box-shadow: 0 20px 40px -10px rgba(4,120,87,0.4);
        border: 1px solid rgba(52,211,153,0.3);
    }
    
    /* Floating metrics */
    .metric-3d {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 20px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid #e2e8f0;
        animation: float 3s ease-in-out infinite;
    }
    
    .metric-3d:hover {
        transform: translateY(-3px);
    }
    
    .metric-value-3d {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    
    .metric-label-3d {
        font-size: 0.7rem;
        color: #64748b;
        margin-top: 0.25rem;
        letter-spacing: 0.5px;
    }
    
    /* 3D Button */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 40px;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        box-shadow: 0 4px 14px 0 rgba(59,130,246,0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(59,130,246,0.5);
    }
    
    /* Text area with 3D effect */
    .stTextArea textarea {
        border-radius: 20px;
        border: 1.5px solid #e2e8f0;
        font-size: 1rem;
        line-height: 1.6;
        transition: all 0.3s ease;
        background: white;
    }
    
    .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.1);
        transform: scale(1.01);
    }
    
    /* Confidence bar 3D */
    .confidence-3d {
        background: #e2e8f0;
        border-radius: 20px;
        height: 12px;
        overflow: hidden;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 20px;
        transition: width 0.8s cubic-bezier(0.34, 1.2, 0.64, 1);
        position: relative;
        overflow: hidden;
    }
    
    .confidence-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: none;
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #94a3b8;
        font-size: 0.7rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
    }
    
    /* Model selector pills */
    .model-pill {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 40px;
        font-size: 0.8rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        background: #f1f5f9;
        color: #1e293b;
    }
    
    .model-pill-active {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(59,130,246,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TEXT CLEANING & HELPERS
# ============================================================================
SLANG = {
    'bc': 'because', 'u': 'you', 'r': 'are', 'w/': 'with',
    'idk': 'i do not know', 'rn': 'right now', 'smh': 'disappointed',
    'lol': '', 'tbh': 'to be honest', 'ngl': 'not going to lie',
    'imo': 'in my opinion', 'btw': 'by the way'
}

CRISIS_KEYWORDS = [
    'suicide', 'kill myself', 'end my life', 'want to die', 
    'no reason to live', 'hopeless', 'worthless', 'cant go on',
    'overdose', 'self harm', 'cutting', 'nothing left', 'give up'
]

def clean_text(text):
    if not text:
        return ""
    text = str(text).lower().strip()
    for k, v in SLANG.items():
        text = re.sub(r'\b' + re.escape(k) + r'\b', v, text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def rule_based_detect(text):
    text_lower = text.lower()
    for kw in CRISIS_KEYWORDS:
        if kw in text_lower:
            return "Crisis", 0.92
    return None, 0.0

# ============================================================================
# LOAD MODELS
# ============================================================================
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'LR': 'saved_models/lr_augmented.pkl',
        'RF': 'saved_models/rf_augmented.pkl',
        'GB': 'saved_models/gb_augmented.pkl'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    
    tfidf = None
    le = None
    
    if os.path.exists('saved_models/tfidf_augmented.pkl'):
        tfidf = joblib.load('saved_models/tfidf_augmented.pkl')
    if os.path.exists('saved_models/label_encoder.pkl'):
        le = joblib.load('saved_models/label_encoder.pkl')
    
    return models, tfidf, le

def predict_text(text, model, tfidf, le):
    if not text:
        return None, 0.0
    
    cleaned = clean_text(text)
    rule_pred, rule_conf = rule_based_detect(cleaned)
    
    try:
        vec = tfidf.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec).max()
        label = le.inverse_transform([pred])[0]
        
        if rule_pred and rule_conf > 0.8:
            return rule_pred, max(rule_conf, proba)
        return label, proba
    except:
        if rule_pred:
            return rule_pred, rule_conf
        return "Neutral", 0.5

# ============================================================================
# LOAD MODELS
# ============================================================================
models, tfidf, le = load_models()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### MODEL")
    
    if models:
        selected_model = st.selectbox("", list(models.keys()))
    else:
        selected_model = None
        st.error("Models not found")
    
    st.markdown("---")
    st.markdown("### CRISIS LINES")
    st.markdown("**USA / CANADA**")
    st.code("988", language=None)
    st.caption("Suicide & Crisis Lifeline")
    
    st.markdown("**TEXT LINE**")
    st.code("Text HOME to 741741", language=None)
    
    st.markdown("**UK**")
    st.code("111", language=None)
    
    st.markdown("**AUSTRALIA**")
    st.code("13 11 14", language=None)

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="header-3d animated-container">
    <h1>MindGuard</h1>
    <p>Real-time mental health text analysis for early crisis detection</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN LAYOUT
# ============================================================================
col_left, col_right = st.columns([2, 1.2])

with col_left:
    st.markdown('<div class="animated-container">', unsafe_allow_html=True)
    st.markdown("### INPUT")
    user_input = st.text_area(
        "",
        height=160,
        placeholder="Paste or type text for analysis...",
        key="input"
    )
    
    analyze = st.button("ANALYZE", use_container_width=True)
    
    st.markdown("### EXAMPLES")
    ex_cols = st.columns(3)
    examples = [
        "I feel hopeless and worthless. Nothing matters anymore.",
        "I'm struggling with anxiety. Does anyone have coping advice?",
        "New study shows meditation reduces depression symptoms by 40%."
    ]
    
    for i, ex in enumerate(examples):
        if ex_cols[i].button(f"Sample {i+1}", use_container_width=True):
            user_input = ex
            analyze = True
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card-3d animated-container">', unsafe_allow_html=True)
    st.markdown("### DETECTION")
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <div style="font-size: 0.7rem; color: #64748b;">CRISIS</div>
        <div style="font-size: 0.8rem;">Suicide, hopelessness, worthlessness, self-harm</div>
    </div>
    <div style="margin-bottom: 1rem;">
        <div style="font-size: 0.7rem; color: #64748b;">SUPPORT</div>
        <div style="font-size: 0.8rem;">Struggling, need help, coping, advice</div>
    </div>
    <div>
        <div style="font-size: 0.7rem; color: #64748b;">NEUTRAL</div>
        <div style="font-size: 0.8rem;">Information, research, general discussion</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# ANALYSIS
# ============================================================================
if analyze and user_input:
    if selected_model and models and tfidf and le:
        
        with st.spinner("Processing..."):
            time.sleep(0.15)
            model = models[selected_model]
            prediction, confidence = predict_text(user_input, model, tfidf, le)
        
        # Result card with 3D animation
        if prediction == "Crisis":
            st.markdown("""
            <div class="result-crisis animated-container">
                <div style="font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem;">CRISIS DETECTED</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">This text contains indicators of a potential mental health crisis.</div>
                <div style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.8;">Immediate support recommended.</div>
            </div>
            """, unsafe_allow_html=True)
        elif prediction == "Support":
            st.markdown("""
            <div class="result-support animated-container">
                <div style="font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem;">SUPPORT INDICATED</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">This text suggests the user may benefit from supportive intervention.</div>
                <div style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.8;">Active listening is recommended.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-neutral animated-container">
                <div style="font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem;">NEUTRAL / INFORMATIONAL</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">This text appears to be neutral or informational in nature.</div>
                <div style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.8;">Continue providing awareness.</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Metrics row with 3D floating effect
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-3d">
                <div class="metric-value-3d">{prediction}</div>
                <div class="metric-label-3d">CLASSIFICATION</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-3d">
                <div class="metric-value-3d">{confidence:.1%}</div>
                <div class="metric-label-3d">CONFIDENCE</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            word_count = len(user_input.split())
            st.markdown(f"""
            <div class="metric-3d">
                <div class="metric-value-3d">{word_count}</div>
                <div class="metric-label-3d">WORDS</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            char_count = len(user_input)
            st.markdown(f"""
            <div class="metric-3d">
                <div class="metric-value-3d">{char_count}</div>
                <div class="metric-label-3d">CHARACTERS</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 3D Confidence bar
        bar_color = "#ef4444" if prediction == "Crisis" else "#3b82f6" if prediction == "Support" else "#10b981"
        st.markdown("### CONFIDENCE")
        st.markdown(f"""
        <div class="confidence-3d">
            <div class="confidence-fill" style="width: {confidence*100}%; background: {bar_color};"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Detected keywords
        detected = []
        text_lower = user_input.lower()
        for kw in CRISIS_KEYWORDS:
            if kw in text_lower:
                detected.append(kw)
        
        if detected:
            st.markdown("### KEY INDICATORS")
            st.markdown(f"<div style='background: #fef2f2; border-radius: 16px; padding: 0.75rem; color: #991b1b; font-size: 0.85rem;'>{', '.join(detected[:6])}</div>", unsafe_allow_html=True)
        
        # Response guide
        st.markdown("### RESPONSE")
        if prediction == "Crisis":
            st.info("Contact a crisis helpline immediately. Ensure the person is not alone.")
        elif prediction == "Support":
            st.info("Listen without judgment. Validate their feelings. Ask how you can help.")
        else:
            st.info("Provide mental health resources. Encourage self-care practices.")
        
    else:
        st.error("Models not available.")

elif analyze and not user_input:
    st.warning("Please enter text for analysis.")

# ============================================================================
# RESOURCES SECTION
# ============================================================================
st.markdown("---")
st.markdown("### HELPLINES")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**United States**")
    st.code("988", language=None)
with col2:
    st.markdown("**Crisis Text Line**")
    st.code("Text HOME to 741741", language=None)
with col3:
    st.markdown("**United Kingdom**")
    st.code("111", language=None)
with col4:
    st.markdown("**Australia**")
    st.code("13 11 14", language=None)

# ============================================================================
# SELF-CARE TECHNIQUES
# ============================================================================
with st.expander("SELF-CARE TECHNIQUES"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**5-4-3-2-1 Grounding**")
        st.caption("5 things you see | 4 you feel | 3 you hear | 2 you smell | 1 you taste")
        
        st.markdown("**Box Breathing**")
        st.caption("Inhale 4s → Hold 4s → Exhale 4s → Hold 4s")
    
    with col2:
        st.markdown("**Progressive Muscle Relaxation**")
        st.caption("Tense and release each muscle group from toes to head")
        
        st.markdown("**Mindfulness**")
        st.caption("Focus on present moment without judgment")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer">
    MindGuard | Real-time mental health crisis detection
</div>
""", unsafe_allow_html=True)
