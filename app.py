# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import time
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Page config
st.set_page_config(
    page_title="MindGuard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .main-header {
        background: #1a1a2e;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        border-bottom: 3px solid #16213e;
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 500;
        letter-spacing: 1px;
    }
    
    .main-header p {
        color: #a0a0a0;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
    
    .result-box {
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .result-crisis {
        background: #dc2626;
        color: white;
        border-left: 4px solid #991b1b;
    }
    
    .result-support {
        background: #3b82f6;
        color: white;
        border-left: 4px solid #1e40af;
    }
    
    .result-neutral {
        background: #10b981;
        color: white;
        border-left: 4px solid #047857;
    }
    
    .metric-card {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
    }
    
    .metric-label {
        font-size: 0.7rem;
        color: #64748b;
        margin-top: 0.25rem;
    }
    
    .confidence-bar {
        background: #e2e8f0;
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    .stButton > button {
        background: #1a1a2e;
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 6px;
        font-weight: 500;
        width: 100%;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: #16213e;
        transform: translateY(-1px);
    }
    
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        font-size: 0.9rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.1);
    }
    
    .footer {
        text-align: center;
        padding: 1rem;
        color: #94a3b8;
        font-size: 0.7rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
    }
    
    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
        padding: 0 1rem;
    }
    
    .step {
        text-align: center;
        flex: 1;
        position: relative;
    }
    
    .step-number {
        width: 30px;
        height: 30px;
        background: #cbd5e1;
        color: white;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .step.active .step-number {
        background: #3b82f6;
    }
    
    .step.completed .step-number {
        background: #10b981;
    }
    
    .step-label {
        font-size: 0.7rem;
        color: #64748b;
    }
    
    .step.active .step-label {
        color: #1e293b;
        font-weight: 500;
    }
    
    hr {
        margin: 1rem 0;
        border-color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Text cleaning
STOP_WORDS = set(stopwords.words('english')) if 'stopwords' in locals() else set()

CRISIS_KEYWORDS = [
    'suicide', 'kill myself', 'end my life', 'want to die', 
    'no reason to live', 'hopeless', 'worthless', 'cant go on',
    'overdose', 'self harm', 'cutting', 'nothing left', 'give up'
]

def clean_text(text):
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_sentiment(text):
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return "Positive", polarity
        elif polarity < -0.1:
            return "Negative", polarity
        else:
            return "Neutral", polarity
    except:
        return "Neutral", 0.0

def rule_based_detect(text):
    text_lower = text.lower()
    for kw in CRISIS_KEYWORDS:
        if kw in text_lower:
            return "Crisis", 0.92
    return None, 0.0

def get_text_stats(text):
    words = text.split()
    chars = len(text)
    return {
        'words': len(words),
        'characters': chars,
        'avg_word_length': chars / max(len(words), 1)
    }

def get_word_frequencies(text):
    words = re.findall(r'\b[a-z]+\b', text.lower())
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return Counter(words).most_common(10)

# Load whisper for voice to text
@st.cache_resource
def load_whisper():
    try:
        import whisper
        return whisper.load_model("base")
    except:
        return None

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'current_text' not in st.session_state:
    st.session_state.current_text = ""
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# Header
st.markdown("""
<div class="main-header">
    <h1>MindGuard</h1>
    <p>Mental Health Text Analysis System</p>
</div>
""", unsafe_allow_html=True)

# Step indicator
st.markdown("""
<div class="step-indicator">
    <div class="step active">
        <div class="step-number">1</div>
        <div class="step-label">Input</div>
    </div>
    <div class="step">
        <div class="step-number">2</div>
        <div class="step-label">Analysis</div>
    </div>
    <div class="step">
        <div class="step-number">3</div>
        <div class="step-label">Results</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main content area
if st.session_state.step == 1:
    st.markdown("### Step 1: Input Text")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Upload Audio File")
        audio_file = st.file_uploader("Upload audio file (MP3, WAV, M4A)", type=['mp3', 'wav', 'm4a'])
        
        if audio_file:
            with st.spinner("Transcribing audio..."):
                whisper_model = load_whisper()
                if whisper_model:
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(audio_file.read())
                        tmp_path = tmp_file.name
                    
                    try:
                        result = whisper_model.transcribe(tmp_path)
                        st.session_state.current_text = result["text"]
                        st.success("Audio transcribed successfully!")
                    except Exception as e:
                        st.error(f"Transcription failed: {str(e)}")
                    
                    os.unlink(tmp_path)
                else:
                    st.info("Voice transcription requires additional setup. Please use text input below.")
    
    with col2:
        st.markdown("#### Or Type/Paste Text")
        user_text = st.text_area(
            "",
            height=150,
            placeholder="Enter text for mental health analysis...",
            key="text_input"
        )
        
        if user_text:
            st.session_state.current_text = user_text
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Continue to Analysis", use_container_width=True) and st.session_state.current_text:
            st.session_state.step = 2
            st.rerun()
        elif st.button("Continue to Analysis", use_container_width=True) and not st.session_state.current_text:
            st.warning("Please provide text input first")

elif st.session_state.step == 2:
    st.markdown("### Step 2: Analyzing Text")
    
    text_to_analyze = st.session_state.current_text
    
    st.markdown("#### Input Text:")
    st.info(text_to_analyze[:500] + ("..." if len(text_to_analyze) > 500 else ""))
    
    with st.spinner("Analyzing..."):
        time.sleep(0.3)
        
        # Rule-based detection
        rule_pred, rule_conf = rule_based_detect(text_to_analyze)
        
        if rule_pred:
            prediction = rule_pred
            confidence = rule_conf
        else:
            # Simple ML alternative
            sentiment, polarity = get_sentiment(text_to_analyze)
            if sentiment == "Negative":
                prediction = "Support"
                confidence = 0.65
            elif sentiment == "Positive":
                prediction = "Neutral"
                confidence = 0.70
            else:
                prediction = "Neutral"
                confidence = 0.60
        
        stats = get_text_stats(text_to_analyze)
        sentiment, polarity = get_sentiment(text_to_analyze)
        
        st.session_state.analysis_result = {
            'prediction': prediction,
            'confidence': confidence,
            'stats': stats,
            'sentiment': sentiment,
            'polarity': polarity,
            'text': text_to_analyze
        }
        
        st.session_state.analysis_complete = True
        st.session_state.step = 3
        st.rerun()

elif st.session_state.step == 3:
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        prediction = result['prediction']
        confidence = result['confidence']
        
        st.markdown("### Step 3: Analysis Results")
        
        # Result box
        if prediction == "Crisis":
            st.markdown(f"""
            <div class="result-box result-crisis">
                <strong>CRISIS DETECTED</strong><br>
                This text contains indicators suggesting immediate attention is needed.
            </div>
            """, unsafe_allow_html=True)
        elif prediction == "Support":
            st.markdown(f"""
            <div class="result-box result-support">
                <strong>SUPPORT INDICATED</strong><br>
                This text suggests the person may benefit from supportive intervention.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box result-neutral">
                <strong>NEUTRAL ASSESSMENT</strong><br>
                This text appears to be neutral or informational in nature.
            </div>
            """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{prediction}</div>
                <div class="metric-label">Classification</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{confidence:.1%}</div>
                <div class="metric-label">Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result['stats']['words']}</div>
                <div class="metric-label">Word Count</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result['sentiment']}</div>
                <div class="metric-label">Sentiment</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Confidence bar
        bar_color = "#dc2626" if prediction == "Crisis" else "#3b82f6" if prediction == "Support" else "#10b981"
        st.markdown(f"""
        <div style="margin: 1rem 0;">
            <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 0.25rem;">Confidence Level</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence*100}%; background: {bar_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Top words
        top_words = get_word_frequencies(result['text'])
        if top_words:
            st.markdown("#### Top Keywords")
            words, counts = zip(*top_words)
            fig = go.Figure(data=[go.Bar(x=list(words), y=list(counts), marker_color='#3b82f6')])
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        
        # Response guide
        st.markdown("#### Recommended Action")
        if prediction == "Crisis":
            st.error("Contact a crisis helpline immediately. Ensure the person is not alone.")
            st.info("Crisis Helpline: 988 (US/Canada)")
        elif prediction == "Support":
            st.warning("Listen without judgment. Validate their feelings. Ask how you can help.")
        else:
            st.success("Continue providing mental health awareness and resources.")
        
        st.markdown("---")
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("< Back to Input", use_container_width=True):
                st.session_state.step = 1
                st.session_state.analysis_complete = False
                st.session_state.current_text = ""
                st.session_state.analysis_result = None
                st.rerun()
        
        with col2:
            if st.button("New Analysis", use_container_width=True):
                st.session_state.step = 1
                st.session_state.analysis_complete = False
                st.session_state.current_text = ""
                st.session_state.analysis_result = None
                st.rerun()
        
        with col3:
            if st.button("Exit System", use_container_width=True):
                st.session_state.step = 1
                st.session_state.analysis_complete = False
                st.session_state.current_text = ""
                st.session_state.analysis_result = None
                st.success("Session cleared. Thank you for using MindGuard.")
                st.rerun()

# Footer
st.markdown("""
<div class="footer">
    MindGuard | Mental Health Analysis System
</div>
""", unsafe_allow_html=True)
