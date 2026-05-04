# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from collections import Counter
import time
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    st.warning("WordCloud library not available. Install with: pip install wordcloud")

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
    
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(30px) scale(0.95); }
        100% { opacity: 1; transform: translateY(0) scale(1); }
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(59,130,246,0.4); }
        70% { box-shadow: 0 0 0 15px rgba(59,130,246,0); }
        100% { box-shadow: 0 0 0 0 rgba(59,130,246,0); }
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .header-3d {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        padding: 2rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 35px -10px rgba(0,0,0,0.2);
        animation: fadeInUp 0.5s ease-out;
    }
    
    .header-3d h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 600;
    }
    
    .header-3d p {
        color: #94a3b8;
        margin-top: 0.5rem;
    }
    
    .card-3d {
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        animation: fadeInUp 0.5s ease-out;
    }
    
    .card-3d:hover {
        transform: translateY(-5px);
    }
    
    .result-crisis {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border-radius: 24px;
        padding: 2rem;
        color: white;
        animation: fadeInUp 0.5s ease-out, pulse 2s infinite;
    }
    
    .result-support {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        border-radius: 24px;
        padding: 2rem;
        color: white;
        animation: fadeInUp 0.5s ease-out;
    }
    
    .result-neutral {
        background: linear-gradient(135deg, #064e3b 0%, #047857 100%);
        border-radius: 24px;
        padding: 2rem;
        color: white;
        animation: fadeInUp 0.5s ease-out;
    }
    
    .metric-3d {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 20px;
        padding: 1rem;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid #e2e8f0;
    }
    
    .metric-value-3d {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    
    .metric-label-3d {
        font-size: 0.65rem;
        color: #64748b;
    }
    
    .confidence-bar {
        background: #e2e8f0;
        border-radius: 20px;
        height: 10px;
        overflow: hidden;
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
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 40px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
    }
    
    .stTextArea textarea {
        border-radius: 20px;
        border: 1.5px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.1);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }
    
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #94a3b8;
        font-size: 0.7rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
    }
    
    .eda-container {
        background: #f8fafc;
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1rem;
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

STOP_WORDS = set(stopwords.words('english')) if 'stopwords' in locals() else set()

CRISIS_KEYWORDS = [
    'suicide', 'kill myself', 'end my life', 'want to die', 
    'no reason to live', 'hopeless', 'worthless', 'cant go on',
    'overdose', 'self harm', 'cutting', 'nothing left', 'give up',
    'goodbye', 'ending it', 'done with life', 'no hope'
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

# ============================================================================
# LOAD MODELS (Traditional)
# ============================================================================
@st.cache_resource
def load_traditional_models():
    models = {}
    model_files = {
        'Logistic Regression': 'saved_models/lr_augmented.pkl',
        'Random Forest': 'saved_models/rf_augmented.pkl',
        'Gradient Boosting': 'saved_models/gb_augmented.pkl'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except:
                pass
    
    tfidf = None
    le = None
    
    if os.path.exists('saved_models/tfidf_augmented.pkl'):
        try:
            tfidf = joblib.load('saved_models/tfidf_augmented.pkl')
        except:
            pass
    if os.path.exists('saved_models/label_encoder.pkl'):
        try:
            le = joblib.load('saved_models/label_encoder.pkl')
        except:
            pass
    
    return models, tfidf, le

# ============================================================================
# LOAD PRETRAINED MODELS (DistilBERT, RoBERTa)
# ============================================================================
@st.cache_resource
def load_pretrained_models():
    models = {}
    
    try:
        from transformers import pipeline
        # DistilBERT for sentiment/classification
        models['DistilBERT'] = pipeline(
            "text-classification", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )
    except Exception as e:
        models['DistilBERT'] = None
    
    try:
        from transformers import pipeline
        # RoBERTa for emotion detection
        models['RoBERTa'] = pipeline(
            "text-classification",
            model="bhadresh-savani/roberta-base-emotion",
            device=-1
        )
    except Exception as e:
        models['RoBERTa'] = None
    
    return models

def predict_with_pretrained(text, model, model_name):
    if not model:
        return "Neutral", 0.5
    
    try:
        result = model(text[:512])[0]
        label = result['label']
        score = result['score']
        
        if model_name == 'DistilBERT':
            if label == 'POSITIVE':
                return "Neutral", score
            else:
                return "Support", score
        else:
            if label in ['anger', 'fear', 'sadness']:
                return "Crisis", score
            elif label in ['joy', 'surprise']:
                return "Neutral", score
            else:
                return "Support", score
    except:
        return "Neutral", 0.5

# ============================================================================
# TEXT ANALYSIS FUNCTIONS
# ============================================================================
def get_word_frequencies(text):
    words = re.findall(r'\b[a-z]+\b', text.lower())
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return Counter(words).most_common(15)

def get_text_stats(text):
    words = text.split()
    chars = len(text)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    
    return {
        'words': len(words),
        'characters': chars,
        'sentences': len(sentences),
        'avg_word_length': chars / max(len(words), 1)
    }

# ============================================================================
# DATA CLEANING & EDA FUNCTIONS
# ============================================================================
def clean_dataframe(df):
    df_clean = df.copy()
    
    if 'text' in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=['text'])
    
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].fillna('Unknown')
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    if 'text' in df_clean.columns:
        df_clean['text_clean'] = df_clean['text'].apply(lambda x: clean_text(str(x)) if pd.notna(x) else '')
        df_clean['text_length'] = df_clean['text_clean'].apply(len)
    
    return df_clean

def create_eda_plots(df):
    plots = []
    
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        fig1 = go.Figure(data=[go.Pie(
            labels=label_counts.index,
            values=label_counts.values,
            marker_colors=['#ef4444', '#3b82f6', '#10b981'],
            hole=0.4
        )])
        fig1.update_layout(title="Label Distribution", height=350)
        plots.append(fig1)
    
    if 'text_length' in df.columns:
        fig2 = go.Figure(data=[go.Histogram(
            x=df['text_length'],
            marker_color='#3b82f6',
            nbinsx=30
        )])
        fig2.update_layout(title="Text Length Distribution", height=350)
        plots.append(fig2)
    
    if 'label' in df.columns and 'text_length' in df.columns:
        fig3 = go.Figure()
        for label in df['label'].unique():
            subset = df[df['label'] == label]['text_length']
            fig3.add_trace(go.Box(y=subset, name=label))
        fig3.update_layout(title="Text Length by Label", height=350)
        plots.append(fig3)
    
    return plots

# ============================================================================
# LOAD MODELS
# ============================================================================
traditional_models, tfidf, le = load_traditional_models()
pretrained_models = load_pretrained_models()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### MODEL SELECTION")
    
    all_models = []
    if traditional_models:
        all_models.extend(list(traditional_models.keys()))
    if pretrained_models.get('DistilBERT'):
        all_models.append('DistilBERT')
    if pretrained_models.get('RoBERTa'):
        all_models.append('RoBERTa')
    
    selected_model = st.selectbox("", all_models if all_models else ["No models available"])
    
    st.markdown("---")
    st.markdown("### CRISIS LINES")
    st.code("988", language=None)
    st.caption("USA / Canada")
    st.code("Text HOME to 741741", language=None)
    st.code("111", language=None)
    st.caption("UK")
    st.code("13 11 14", language=None)
    st.caption("Australia")

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="header-3d">
    <h1>MindGuard</h1>
    <p>Real-time mental health text analysis with multi-model comparison</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["ANALYZE", "DATA CLEANING", "EDA", "MODEL COMPARISON"])

# ============================================================================
# TAB 1: ANALYZE
# ============================================================================
with tab1:
    col_left, col_right = st.columns([2, 1.2])
    
    with col_left:
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
    
    with col_right:
        st.markdown("### DETECTION RULES")
        st.markdown("""
        <div style="background: #f1f5f9; border-radius: 16px; padding: 1rem;">
            <div style="margin-bottom: 1rem;">
                <div style="font-size: 0.7rem; color: #ef4444;">CRISIS INDICATORS</div>
                <div style="font-size: 0.8rem;">Suicide, hopelessness, worthlessness, self-harm</div>
            </div>
            <div style="margin-bottom: 1rem;">
                <div style="font-size: 0.7rem; color: #3b82f6;">SUPPORT INDICATORS</div>
                <div style="font-size: 0.8rem;">Struggling, need help, coping, advice</div>
            </div>
            <div>
                <div style="font-size: 0.7rem; color: #10b981;">NEUTRAL INDICATORS</div>
                <div style="font-size: 0.8rem;">Information, research, general discussion</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if analyze and user_input:
        if selected_model and selected_model != "No models available":
            
            with st.spinner("Processing..."):
                time.sleep(0.15)
                
                if selected_model in traditional_models:
                    model = traditional_models[selected_model]
                    prediction, confidence = predict_text(user_input, model, tfidf, le)
                else:
                    pt_model = pretrained_models.get(selected_model)
                    prediction, confidence = predict_with_pretrained(user_input, pt_model, selected_model)
            
            if prediction == "Crisis":
                st.markdown("""
                <div class="result-crisis">
                    <div style="font-size: 1.3rem; font-weight: 600;">CRISIS DETECTED</div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">This text contains indicators of a potential mental health crisis.</div>
                    <div style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.8;">Immediate support recommended.</div>
                </div>
                """, unsafe_allow_html=True)
            elif prediction == "Support":
                st.markdown("""
                <div class="result-support">
                    <div style="font-size: 1.3rem; font-weight: 600;">SUPPORT INDICATED</div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">This text suggests the user may benefit from supportive intervention.</div>
                    <div style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.8;">Active listening is recommended.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-neutral">
                    <div style="font-size: 1.3rem; font-weight: 600;">NEUTRAL / INFORMATIONAL</div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">This text appears to be neutral or informational in nature.</div>
                    <div style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.8;">Continue providing awareness.</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Metrics
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
                stats = get_text_stats(user_input)
                st.markdown(f"""
                <div class="metric-3d">
                    <div class="metric-value-3d">{stats['words']}</div>
                    <div class="metric-label-3d">WORDS</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-3d">
                    <div class="metric-value-3d">{stats['characters']}</div>
                    <div class="metric-label-3d">CHARACTERS</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence bar
            bar_color = "#ef4444" if prediction == "Crisis" else "#3b82f6" if prediction == "Support" else "#10b981"
            st.markdown("### CONFIDENCE LEVEL")
            st.markdown(f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence*100}%; background: {bar_color};"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Sentiment analysis
            sentiment, polarity = get_sentiment(user_input)
            st.markdown(f"""
            <div class="metric-3d" style="margin-top: 0.5rem;">
                <div class="metric-value-3d">{sentiment}</div>
                <div class="metric-label-3d">SENTIMENT (Polarity: {polarity:.2f})</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Top words
            st.markdown("### TOP FREQUENT WORDS")
            top_words = get_word_frequencies(user_input)
            if top_words:
                words, counts = zip(*top_words)
                fig = go.Figure(data=[go.Bar(x=list(words), y=list(counts), marker_color='#3b82f6')])
                fig.update_layout(height=300, xaxis_title="Word", yaxis_title="Frequency")
                st.plotly_chart(fig, use_container_width=True)
            
            # Response guide
            st.markdown("### RESPONSE GUIDE")
            if prediction == "Crisis":
                st.info("Contact a crisis helpline immediately. Ensure the person is not alone.")
            elif prediction == "Support":
                st.info("Listen without judgment. Validate their feelings. Ask how you can help.")
            else:
                st.info("Provide mental health resources. Encourage self-care practices.")
            
        else:
            st.error("Models not available.")

# ============================================================================
# TAB 2: DATA CLEANING
# ============================================================================
with tab2:
    st.markdown("### DATA CLEANING")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"Original shape: {df.shape}")
        
        st.markdown("#### Original Data Sample")
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("APPLY CLEANING", use_container_width=True):
            with st.spinner("Cleaning data..."):
                time.sleep(0.3)
                df_clean = clean_dataframe(df)
            
            st.success(f"Cleaned shape: {df_clean.shape}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Missing Values (Original)")
                missing_orig = df.isnull().sum()
                missing_orig = missing_orig[missing_orig > 0]
                if len(missing_orig) > 0:
                    st.dataframe(pd.DataFrame({'Missing': missing_orig}))
                else:
                    st.write("No missing values")
            
            with col2:
                st.markdown("#### Missing Values (Cleaned)")
                missing_clean = df_clean.isnull().sum()
                missing_clean = missing_clean[missing_clean > 0]
                if len(missing_clean) > 0:
                    st.dataframe(pd.DataFrame({'Missing': missing_clean}))
                else:
                    st.write("No missing values")
            
            st.markdown("#### Duplicates Removed")
            st.write(f"Original duplicates: {df.duplicated().sum()}")
            st.write(f"After cleaning: {df_clean.duplicated().sum()}")
            
            st.markdown("#### Cleaned Data Sample")
            st.dataframe(df_clean.head(), use_container_width=True)
            
            csv = df_clean.to_csv(index=False)
            st.download_button("DOWNLOAD CLEANED DATA", csv, "cleaned_data.csv", "text/csv")

# ============================================================================
# TAB 3: EDA
# ============================================================================
with tab3:
    st.markdown("### EXPLORATORY DATA ANALYSIS")
    
    eda_file = st.file_uploader("Upload CSV for EDA", type=['csv'], key="eda")
    
    if eda_file:
        df_eda = pd.read_csv(eda_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", df_eda.shape[0])
        with col2:
            st.metric("Columns", df_eda.shape[1])
        
        st.markdown("#### Data Types")
        dtype_df = pd.DataFrame(df_eda.dtypes.reset_index())
        dtype_df.columns = ['Column', 'Data Type']
        st.dataframe(dtype_df, use_container_width=True)
        
        numeric_cols = df_eda.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.markdown("#### Descriptive Statistics")
            st.dataframe(df_eda[numeric_cols].describe(), use_container_width=True)
        
        st.markdown("#### Visualizations")
        plots = create_eda_plots(df_eda)
        
        for plot in plots:
            st.plotly_chart(plot, use_container_width=True)
        
        if len(numeric_cols) >= 2:
            st.markdown("#### Correlation Matrix")
            corr = df_eda[numeric_cols].corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='Blues',
                text=corr.values.round(2),
                texttemplate='%{text}'
            ))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 4: MODEL COMPARISON
# ============================================================================
with tab4:
    st.markdown("### MODEL COMPARISON")
    
    comparison_text = st.text_area(
        "Enter text to compare across all models",
        height=100,
        placeholder="Paste text to see how each model classifies it...",
        key="compare_input"
    )
    
    if st.button("COMPARE MODELS", use_container_width=True) and comparison_text:
        
        results = []
        
        for name, model in traditional_models.items():
            pred, conf = predict_text(comparison_text, model, tfidf, le)
            results.append({
                'Model': name,
                'Type': 'Traditional ML',
                'Prediction': pred,
                'Confidence': f"{conf:.1%}"
            })
        
        for name, model in pretrained_models.items():
            if model:
                pred, conf = predict_with_pretrained(comparison_text, model, name)
                results.append({
                    'Model': name,
                    'Type': 'Deep Learning',
                    'Prediction': pred,
                    'Confidence': f"{conf:.1%}"
                })
        
        rule_pred, rule_conf = rule_based_detect(comparison_text)
        results.append({
            'Model': 'Rule-Based',
            'Type': 'Heuristic',
            'Prediction': rule_pred if rule_pred else "Neutral",
            'Confidence': f"{rule_conf:.1%}" if rule_pred else "N/A"
        })
        
        results_df = pd.DataFrame(results)
        
        def color_pred(val):
            if val == 'Crisis':
                return 'color: #ef4444; font-weight: bold'
            elif val == 'Support':
                return 'color: #3b82f6; font-weight: bold'
            else:
                return 'color: #10b981; font-weight: bold'
        
        styled_df = results_df.style.applymap(color_pred, subset=['Prediction'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        st.markdown("#### Agreement Analysis")
        predictions = results_df['Prediction'].tolist()
        majority = max(set(predictions), key=predictions.count)
        agreement = predictions.count(majority) / len(predictions)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Majority Prediction", majority)
        with col2:
            st.metric("Agreement Rate", f"{agreement:.0%}")
        
        fig = go.Figure(data=[
            go.Bar(
                x=results_df['Model'],
                y=[float(r['Confidence'].replace('%', '')) for r in results if r['Confidence'] != 'N/A'],
                marker_color=['#ef4444' if r['Prediction'] == 'Crisis' else '#3b82f6' if r['Prediction'] == 'Support' else '#10b981' for r in results if r['Confidence'] != 'N/A'],
                text=[r['Prediction'] for r in results if r['Confidence'] != 'N/A'],
                textposition='outside'
            )
        ])
        fig.update_layout(
            title="Model Confidence Comparison",
            xaxis_title="Model",
            yaxis_title="Confidence (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer">
    MindGuard | Multi-model mental health crisis detection
</div>
""", unsafe_allow_html=True)
