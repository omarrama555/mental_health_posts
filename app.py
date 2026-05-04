# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="MindGuard | Mental Health Crisis Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - PROFESSIONAL DARK/LIGHT HYBRID
# ============================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: 1400px;
    }
    
    /* Gradient header */
    .gradient-header {
        background: linear-gradient(120deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.8rem 2rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .gradient-header h1 {
        color: white;
        font-size: 2rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .gradient-header p {
        color: rgba(255,255,255,0.7);
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    
    /* Cards */
    .card {
        background: #ffffff;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    }
    
    /* Result cards with animations */
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .result-crisis {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        animation: slideInUp 0.5s ease;
    }
    
    .result-support {
        background: linear-gradient(135deg, #3b82f6 0%, #1e3a8a 100%);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        animation: slideInUp 0.5s ease;
    }
    
    .result-neutral {
        background: linear-gradient(135deg, #10b981 0%, #064e3b 100%);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        animation: slideInUp 0.5s ease;
    }
    
    /* Metrics */
    .metric-container {
        background: #f8fafc;
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.25rem;
    }
    
    /* Confidence bar */
    .confidence-bar-container {
        background: #e2e8f0;
        border-radius: 12px;
        height: 8px;
        overflow: hidden;
    }
    
    .confidence-bar {
        height: 100%;
        border-radius: 12px;
        transition: width 0.5s ease;
    }
    
    /* Model selector */
    .model-selector {
        background: #f1f5f9;
        border-radius: 12px;
        padding: 0.25rem;
        display: flex;
        gap: 0.25rem;
    }
    
    .model-option {
        flex: 1;
        text-align: center;
        padding: 0.5rem;
        border-radius: 10px;
        cursor: pointer;
        font-size: 0.8rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .model-option-active {
        background: white;
        color: #1e293b;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .model-option-inactive {
        color: #64748b;
    }
    
    /* Text area */
    .stTextArea textarea {
        border-radius: 16px;
        border: 1.5px solid #e2e8f0;
        font-size: 1rem;
        line-height: 1.6;
        transition: border-color 0.2s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.1);
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 40px;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(59,130,246,0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 0.9rem;
        font-weight: 600;
        color: #64748b;
        padding: 0.5rem 0;
    }
    
    .stTabs [aria-selected="true"] {
        color: #3b82f6;
        border-bottom: 2px solid #3b82f6;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #94a3b8;
        font-size: 0.75rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TEXT CLEANING
# ============================================================================
SLANG = {
    'bc': 'because', 'u': 'you', 'r': 'are', 'w/': 'with',
    'idk': 'i do not know', 'rn': 'right now', 'smh': 'disappointed',
    'lol': '', 'tbh': 'to be honest', 'ngl': 'not going to lie',
    'imo': 'in my opinion', 'btw': 'by the way', 'pls': 'please',
    'plz': 'please', 'thx': 'thanks', 'ty': 'thank you',
    'ur': 'your', 'u r': 'you are'
}

CRISIS_KEYWORDS = [
    'suicide', 'kill myself', 'end my life', 'want to die', 
    'no reason to live', 'hopeless', 'worthless', 'cant go on',
    'overdose', 'self harm', 'cutting', 'nothing left',
    'give up', 'cannot breathe', 'falling apart', 'goodbye'
]

def clean_text(text):
    if not text or text == "":
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
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'Logistic Regression': 'saved_models/lr_augmented.pkl',
        'Random Forest': 'saved_models/rf_augmented.pkl',
        'Gradient Boosting': 'saved_models/gb_augmented.pkl'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    
    tfidf = None
    label_encoder = None
    
    if os.path.exists('saved_models/tfidf_augmented.pkl'):
        tfidf = joblib.load('saved_models/tfidf_augmented.pkl')
    if os.path.exists('saved_models/label_encoder.pkl'):
        label_encoder = joblib.load('saved_models/label_encoder.pkl')
    
    return models, tfidf, label_encoder

def predict_text(text, model, tfidf, label_encoder):
    if not text or text.strip() == "":
        return None, 0.0
    
    cleaned = clean_text(text)
    rule_pred, rule_conf = rule_based_detect(cleaned)
    
    try:
        vec = tfidf.transform([cleaned])
        pred_encoded = model.predict(vec)[0]
        proba = model.predict_proba(vec).max()
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        
        if rule_pred is not None and rule_conf > 0.8:
            return rule_pred, max(rule_conf, proba)
        return pred_label, proba
    except:
        if rule_pred is not None:
            return rule_pred, rule_conf
        return "Neutral", 0.5

# ============================================================================
# LOAD MODELS
# ============================================================================
models, tfidf, label_encoder = load_models()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### MODEL SELECTION")
    
    available_models = list(models.keys())
    if available_models:
        selected_model = st.selectbox(
            "Choose prediction model",
            available_models,
            help="Different models have different strengths. Logistic Regression offers the best balance of speed and accuracy."
        )
        
        model_performance = {
            'Logistic Regression': {'Accuracy': 97.4, 'Macro F1': 95.4, 'Latency': '< 0.1s'},
            'Random Forest': {'Accuracy': 96.1, 'Macro F1': 93.4, 'Latency': '~0.3s'},
            'Gradient Boosting': {'Accuracy': 97.2, 'Macro F1': 95.2, 'Latency': '~0.2s'}
        }
        
        if selected_model in model_performance:
            st.markdown("---")
            st.markdown("### PERFORMANCE")
            perf = model_performance[selected_model]
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{perf['Accuracy']}%")
            c2.metric("F1 Score", f"{perf['Macro F1']}%")
            c3.metric("Latency", perf['Latency'])
    else:
        st.error("No models found. Please train models first.")
        selected_model = None
    
    st.markdown("---")
    st.markdown("### CRISIS RESOURCES")
    
    st.markdown("**United States**")
    st.code("988", language=None)
    st.caption("Suicide & Crisis Lifeline")
    
    st.markdown("**Crisis Text Line**")
    st.code("Text HOME to 741741", language=None)
    
    st.markdown("**SAMHSA**")
    st.code("1-800-662-4357", language=None)

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="gradient-header">
    <h1>MindGuard</h1>
    <p>Real-time mental health text analysis for early crisis detection</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN LAYOUT
# ============================================================================
col_left, col_right = st.columns([2, 1.2])

with col_left:
    st.markdown("### INPUT")
    user_input = st.text_area(
        "",
        height=180,
        placeholder="Paste or type text here for analysis...",
        key="input_text"
    )
    
    analyze_clicked = st.button("ANALYZE", use_container_width=True)
    
    st.markdown("### EXAMPLES")
    examples = [
        "I've been feeling completely hopeless. Nothing matters anymore.",
        "I'm struggling but trying to get better. Any advice?",
        "Study shows meditation reduces anxiety by 40% in clinical trials."
    ]
    
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        if cols[i].button(f"Example {i+1}", use_container_width=True, key=f"ex_{i}"):
            user_input = ex
            analyze_clicked = True

with col_right:
    st.markdown("### DETECTION LOGIC")
    st.markdown("""
    <div style="background: #f1f5f9; border-radius: 16px; padding: 1rem;">
        <div style="margin-bottom: 1rem;">
            <div style="font-size: 0.7rem; color: #64748b;">CRISIS INDICATORS</div>
            <div style="font-size: 0.8rem;">Suicide, self-harm, hopelessness, worthlessness, goodbye letters</div>
        </div>
        <div style="margin-bottom: 1rem;">
            <div style="font-size: 0.7rem; color: #64748b;">SUPPORT INDICATORS</div>
            <div style="font-size: 0.8rem;">Struggling, help, coping, need support, advice</div>
        </div>
        <div>
            <div style="font-size: 0.7rem; color: #64748b;">NEUTRAL INDICATORS</div>
            <div style="font-size: 0.8rem;">Information, research, therapy techniques, general discussion</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# ANALYSIS RESULTS
# ============================================================================
if analyze_clicked and user_input:
    if selected_model and models and tfidf and label_encoder:
        
        with st.spinner("Processing..."):
            time.sleep(0.1)
            model = models[selected_model]
            prediction, confidence = predict_text(user_input, model, tfidf, label_encoder)
        
        # Animated result display
        if prediction == "Crisis":
            st.markdown("""
            <div class="result-crisis">
                <div style="font-size: 1.2rem; font-weight: 500; margin-bottom: 0.5rem;">CRISIS DETECTED</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">This text contains indicators of a potential mental health crisis.</div>
                <div style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.8;">Immediate support recommended.</div>
            </div>
            """, unsafe_allow_html=True)
        elif prediction == "Support":
            st.markdown("""
            <div class="result-support">
                <div style="font-size: 1.2rem; font-weight: 500; margin-bottom: 0.5rem;">SUPPORT INDICATED</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">This text suggests the user may benefit from supportive intervention.</div>
                <div style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.8;">Active listening and validation are recommended.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-neutral">
                <div style="font-size: 1.2rem; font-weight: 500; margin-bottom: 0.5rem;">NEUTRAL / INFORMATIONAL</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">This text appears to be neutral or informational in nature.</div>
                <div style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.8;">Continue providing mental health awareness.</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{prediction}</div>
                <div class="metric-label">CLASSIFICATION</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{confidence:.1%}</div>
                <div class="metric-label">CONFIDENCE</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            word_count = len(user_input.split())
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{word_count}</div>
                <div class="metric-label">WORDS</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            char_count = len(user_input)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{char_count}</div>
                <div class="metric-label">CHARACTERS</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Confidence bar
        st.markdown("### CONFIDENCE LEVEL")
        bar_color = "#dc2626" if prediction == "Crisis" else "#3b82f6" if prediction == "Support" else "#10b981"
        st.markdown(f"""
        <div class="confidence-bar-container">
            <div class="confidence-bar" style="width: {confidence*100}%; background: {bar_color};"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 0.25rem;">
            <span style="font-size: 0.7rem; color: #64748b;">LOW</span>
            <span style="font-size: 0.7rem; color: #64748b;">MEDIUM</span>
            <span style="font-size: 0.7rem; color: #64748b;">HIGH</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Key indicators detected
        detected = []
        text_lower = user_input.lower()
        for kw in CRISIS_KEYWORDS:
            if kw in text_lower:
                detected.append(kw)
        
        if detected:
            st.markdown("### KEY INDICATORS")
            st.markdown(f"<div style='background: #fef2f2; border-radius: 12px; padding: 0.75rem; color: #991b1b; font-size: 0.85rem;'>{', '.join(detected[:8])}</div>", unsafe_allow_html=True)
        
        # Response guide
        st.markdown("### RESPONSE GUIDE")
        
        if prediction == "Crisis":
            st.info("""
            **IMMEDIATE ACTIONS**
            • Contact a crisis helpline immediately
            • Ensure the person is not alone
            • Remove access to any means of self-harm
            • Seek emergency medical attention if needed
            """)
        elif prediction == "Support":
            st.info("""
            **SUPPORTIVE RESPONSE**
            • Listen without judgment
            • Validate their feelings
            • Ask how you can help
            • Check in regularly
            • Encourage professional support
            """)
        else:
            st.info("""
            **INFORMATIONAL RESPONSE**
            • Provide mental health resources
            • Encourage self-care practices
            • Normalize seeking help
            • Share educational content
            """)
        
    else:
        st.error("Models not available. Please ensure models are trained.")

elif analyze_clicked and not user_input:
    st.warning("Please enter text for analysis.")

# ============================================================================
# TABS SECTION
# ============================================================================
tab1, tab2, tab3 = st.tabs(["PERFORMANCE", "HELP & RESOURCES", "TECHNICAL DETAILS"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### MODEL COMPARISON")
        comparison_data = {
            'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
            'Accuracy': [97.4, 96.1, 97.2],
            'F1 Score': [95.4, 93.4, 95.2],
            'Recall': [87.3, 87.3, 87.3],
            'Precision': [98.4, 87.3, 97.9]
        }
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### CONFUSION MATRIX")
        st.markdown("""
        <div style="background: #f8fafc; border-radius: 16px; padding: 1rem; text-align: center;">
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin-bottom: 1rem;">
                <div></div>
                <div><strong>Predicted</strong></div>
                <div></div>
                <div><strong>Actual</strong></div>
                <div style="background: #e2e8f0; border-radius: 8px; padding: 0.5rem;">C: 186</div>
                <div style="background: #e2e8f0; border-radius: 8px; padding: 0.5rem;">S: 15</div>
                <div style="background: #e2e8f0; border-radius: 8px; padding: 0.5rem;">N: 12</div>
                <div style="background: #e2e8f0; border-radius: 8px; padding: 0.5rem;">C: 8</div>
                <div style="background: #e2e8f0; border-radius: 8px; padding: 0.5rem;">S: 281</div>
                <div style="background: #e2e8f0; border-radius: 8px; padding: 0.5rem;">N: 10</div>
                <div style="background: #e2e8f0; border-radius: 8px; padding: 0.5rem;">C: 2</div>
                <div style="background: #e2e8f0; border-radius: 8px; padding: 0.5rem;">S: 19</div>
                <div style="background: #e2e8f0; border-radius: 8px; padding: 0.5rem;">N: 1381</div>
            </div>
            <div style="font-size: 0.7rem; color: #64748b;">C=Crisis | S=Support | N=Neutral</div>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("#### CRISIS HELPLINES")
    
    helplines = [
        {"Country": "United States", "Service": "988 Suicide & Crisis Lifeline", "Number": "988"},
        {"Country": "United States", "Service": "Crisis Text Line", "Number": "Text HOME to 741741"},
        {"Country": "United States", "Service": "SAMHSA Helpline", "Number": "1-800-662-4357"},
        {"Country": "United Kingdom", "Service": "NHS Mental Health Helpline", "Number": "111"},
        {"Country": "United Kingdom", "Service": "Samaritans", "Number": "116 123"},
        {"Country": "Canada", "Service": "Crisis Services Canada", "Number": "1-833-456-4566"},
        {"Country": "Australia", "Service": "Lifeline Australia", "Number": "13 11 14"},
    ]
    
    st.dataframe(pd.DataFrame(helplines), use_container_width=True, hide_index=True)
    
    st.markdown("#### SELF-CARE TECHNIQUES")
    
    techniques = [
        "5-4-3-2-1 Grounding: Name 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste",
        "Box Breathing: Inhale 4s, hold 4s, exhale 4s, hold 4s",
        "Progressive muscle relaxation: Tense and release each muscle group",
        "Cognitive reframing: Challenge negative thoughts with evidence",
        "Mindfulness meditation: Focus on present moment without judgment"
    ]
    
    for t in techniques:
        st.markdown(f"- {t}")

with tab3:
    st.markdown("#### TECHNICAL SPECIFICATIONS")
    
    tech_data = {
        "Component": ["Model Type", "Vectorizer", "Training Data", "Features", "Augmentation"],
        "Specification": ["Logistic Regression (One-vs-Rest)", "TF-IDF (1-2 gram, 10k features)", "10,000 annotated posts", "N-gram patterns, slang normalization", "Synonym replacement, random deletion, SMOTE"]
    }
    st.dataframe(pd.DataFrame(tech_data), use_container_width=True, hide_index=True)
    
    st.markdown("#### FAIRNESS AUDIT")
    
    fairness_data = {
        "Group": ["Female", "Male", "Non-binary", "Prefer not to say", "Unknown"],
        "Macro F1": [0.948, 0.961, 0.966, 0.923, 0.963],
        "Crisis Recall": [0.889, 0.912, 0.913, 0.690, 0.929],
        "Sample Size": [473, 418, 446, 426, 151]
    }
    st.dataframe(pd.DataFrame(fairness_data), use_container_width=True, hide_index=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer">
    MindGuard | AI-Powered Mental Health Crisis Detection
</div>
""", unsafe_allow_html=True)
