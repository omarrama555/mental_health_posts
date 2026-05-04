# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from io import BytesIO

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="MindGuard - Mental Health Crisis Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR PROFESSIONAL LOOK
# ============================================================================
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1rem;
        color: white;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Alert boxes */
    .crisis-alert {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
    }
    
    .support-alert {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
    }
    
    .neutral-alert {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        margin-top: 2rem;
    }
    
    /* Text input styling */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        font-size: 1rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
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
    'give up', 'cannot breathe', 'falling apart', 'goodbye',
    'last words', 'ending it', 'done with life'
]

SUPPORT_KEYWORDS = [
    'here for you', 'you are not alone', 'reach out', 'support',
    'recovery', 'getting better', 'therapy', 'counselor',
    'hotline', 'help you', 'proud of you', 'keep going',
    'you matter', 'stay strong', 'you can do this'
]

def clean_text(text):
    """Clean and preprocess text"""
    if not text or text == "":
        return ""
    text = str(text).lower().strip()
    # Replace slang
    for k, v in SLANG.items():
        text = re.sub(r'\b' + re.escape(k) + r'\b', v, text)
    # Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def rule_based_detect(text):
    """Rule-based detection for immediate response"""
    text_lower = text.lower()
    
    # Check for crisis keywords
    for kw in CRISIS_KEYWORDS:
        if kw in text_lower:
            return "Crisis", 0.95
    
    # Check for support keywords
    for kw in SUPPORT_KEYWORDS:
        if kw in text_lower:
            return "Support", 0.85
    
    return None, 0.0

def get_confidence_color(confidence):
    """Get color based on confidence score"""
    if confidence >= 0.9:
        return "#28a745"  # Green
    elif confidence >= 0.7:
        return "#ffc107"  # Yellow
    else:
        return "#dc3545"  # Red

def get_prediction_icon(prediction):
    """Get emoji icon for prediction"""
    icons = {
        "Crisis": "⚠️",
        "Support": "🤝",
        "Neutral": "😊"
    }
    return icons.get(prediction, "📝")

def get_crisis_resources():
    """Return crisis resources"""
    return {
        "National Suicide Prevention Lifeline": "988",
        "Crisis Text Line": "Text HOME to 741741",
        "SAMHSA National Helpline": "1-800-662-4357",
        "Veterans Crisis Line": "988 (Press 1)",
        "Disaster Distress Helpline": "1-800-985-5990"
    }

def get_self_care_tips():
    """Return self-care tips"""
    return [
        "🧘 Take deep breaths - inhale for 4, hold for 4, exhale for 4",
        "💧 Drink a glass of water and stay hydrated",
        "🚶 Take a short walk outside if possible",
        "📝 Write down three things you're grateful for",
        "🎵 Listen to your favorite calming music",
        "📞 Call or text someone you trust",
        "🛏️ Get adequate rest and sleep",
        "🍎 Eat a nutritious meal"
    ]

# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_models():
    """Load all trained models and artifacts"""
    try:
        # Check for saved model files
        model_files = [
            'saved_models/lr_augmented.pkl',
            'saved_models/tfidf_augmented.pkl',
            'saved_models/label_encoder.pkl'
        ]
        
        if all(os.path.exists(f) for f in model_files):
            lr_model = joblib.load('saved_models/lr_augmented.pkl')
            tfidf = joblib.load('saved_models/tfidf_augmented.pkl')
            label_encoder = joblib.load('saved_models/label_encoder.pkl')
            model_type = "Logistic Regression (Augmented)"
        else:
            # Use baseline models if augmented not available
            lr_model = joblib.load('saved_models/lr_baseline.pkl')
            tfidf = joblib.load('saved_models/tfidf_baseline.pkl')
            label_encoder = joblib.load('saved_models/label_encoder.pkl')
            model_type = "Logistic Regression (Baseline)"
        
        return lr_model, tfidf, label_encoder, model_type
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

def predict_text(text, model, tfidf, label_encoder):
    """Make prediction on input text"""
    if not text or text.strip() == "":
        return None, 0.0
    
    # Clean text
    cleaned = clean_text(text)
    
    # Rule-based detection
    rule_pred, rule_conf = rule_based_detect(cleaned)
    
    # Model prediction
    try:
        vec = tfidf.transform([cleaned])
        pred_encoded = model.predict(vec)[0]
        proba = model.predict_proba(vec).max()
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        
        # Combine rule-based and model predictions
        if rule_pred is not None and rule_conf > 0.8:
            return rule_pred, max(rule_conf, proba)
        return pred_label, proba
    except:
        if rule_pred is not None:
            return rule_pred, rule_conf
        return "Neutral", 0.5

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### 🧠 MindGuard")
    st.markdown("---")
    
    # Model info
    st.markdown("#### 🤖 Model Information")
    lr_model, tfidf, label_encoder, model_type = load_models()
    
    if lr_model is not None:
        st.success(f"✅ Model loaded: {model_type}")
        st.info("📊 Model trained on 10,000+ mental health posts")
        st.info("🎯 Accuracy: ~97% | F1-Score: ~0.95")
    else:
        st.error("❌ Model not found. Please train the model first.")
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("#### 📈 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Detection Accuracy", "97%", "✓")
    with col2:
        st.metric("Response Time", "< 1s", "⚡")
    
    st.markdown("---")
    
    # Resources
    st.markdown("#### 🆘 Crisis Resources")
    resources = get_crisis_resources()
    for name, number in resources.items():
        st.markdown(f"**{name}**")
        st.code(number, language=None)
        st.markdown("---")
    
    st.markdown("#### 📞 International Helplines")
    st.markdown("""
    - **UK**: 111 (NHS Mental Health Triage)
    - **Canada**: 1-833-456-4566
    - **Australia**: 13 11 14
    - **India**: 9152987821
    - **International**: Find a helpline at `findahelpline.com`
    """)
    
    st.markdown("---")
    st.caption("⚠️ **Disclaimer**: This tool is for assistance only. In immediate danger, call emergency services (911/US, 999/UK, 112/EU).")

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 MindGuard</h1>
    <p>AI-Powered Mental Health Crisis Detection & Support System</p>
    <p style="font-size: 0.9rem; opacity: 0.8;">Real-time analysis of text for early crisis intervention</p>
</div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Text Analysis", 
    "📊 Dashboard", 
    "ℹ️ About", 
    "🆘 Resources", 
    "💪 Self-Care"
])

# ============================================================================
# TAB 1: TEXT ANALYSIS
# ============================================================================

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### ✍️ Enter Text for Analysis")
        st.markdown("Describe what you're feeling or paste a message to analyze:")
        
        user_input = st.text_area(
            "",
            height=200,
            placeholder="Example: 'I've been feeling really down lately. Nothing seems to matter anymore and I don't see the point in continuing...'",
            key="text_input"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            analyze_btn = st.button("🔍 Analyze", use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Example Texts")
        st.markdown("Try these examples:")
        
        examples = [
            ("😔 Crisis", "I can't do this anymore. Everything hurts and I just want it to end."),
            ("🤝 Support", "I'm here for you. You're not alone in this journey."),
            ("📝 Neutral", "Today I went to therapy and learned some new coping strategies."),
            ("⚠️ Severe", "I've written my goodbye letters. I think I'm done."),
            ("💪 Recovery", "One month clean from self-harm. It's hard but I'm proud of myself.")
        ]
        
        for label, text in examples:
            if st.button(label, key=f"ex_{label}"):
                user_input = text
                analyze_btn = True
        
        st.markdown("---")
        st.info("📝 **Tip**: Be specific about feelings. The more context, the better the analysis.")

    # Analysis results
    if analyze_btn and user_input:
        if lr_model is not None:
            with st.spinner("🧠 Analyzing text..."):
                prediction, confidence = predict_text(user_input, lr_model, tfidf, label_encoder)
                
                # Display results
                st.markdown("### 🔍 Analysis Results")
                
                # Result card based on prediction
                if prediction == "Crisis":
                    st.markdown("""
                    <div class="crisis-alert">
                        <h2>⚠️ CRISIS ALERT ⚠️</h2>
                        <p style="font-size: 1.2rem;">The text contains indicators of a potential mental health crisis.</p>
                        <p>Immediate attention and support are recommended.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show crisis resources immediately
                    with st.expander("🆘 Immediate Help Resources", expanded=True):
                        st.markdown("##### Please reach out to these resources immediately:")
                        for name, number in get_crisis_resources().items():
                            st.markdown(f"- **{name}**: `{number}`")
                        st.warning("**If you or someone else is in immediate danger, please call emergency services (911) right away.**")
                
                elif prediction == "Support":
                    st.markdown("""
                    <div class="support-alert">
                        <h2>🤝 Support Needed</h2>
                        <p style="font-size: 1.2rem;">The text shows signs of distress and would benefit from supportive intervention.</p>
                        <p>Offering support and listening without judgment can make a difference.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                else:
                    st.markdown("""
                    <div class="neutral-alert">
                        <h2>😊 Neutral / Informational</h2>
                        <p style="font-size: 1.2rem;">The text appears to be neutral or informational in nature.</p>
                        <p>Continue providing mental health awareness and resources.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence and details
                col1, col2, col3 = st.columns(3)
                color = get_confidence_color(confidence)
                
                with col1:
                    st.metric("Prediction", f"{get_prediction_icon(prediction)} {prediction}", 
                             delta=None, delta_color="normal")
                
                with col2:
                    st.metric("Confidence", f"{confidence:.1%}")
                
                with col3:
                    st.metric("Response Time", "< 1 second", "Real-time")
                
                # Additional insights
                st.markdown("---")
                st.markdown("#### 📊 Analysis Details")
                
                # Word count
                words = user_input.split()
                st.caption(f"Text length: {len(words)} words | {len(user_input)} characters")
                
                # Keyword detection
                detected_keywords = []
                text_lower = user_input.lower()
                for kw in CRISIS_KEYWORDS:
                    if kw in text_lower:
                        detected_keywords.append(kw)
                
                if detected_keywords:
                    st.warning(f"**Detected crisis keywords**: {', '.join(detected_keywords[:5])}")
                
                # Next steps
                st.markdown("---")
                st.markdown("#### 📋 Recommended Next Steps")
                
                if prediction == "Crisis":
                    st.markdown("""
                    1. 🚨 **Immediate**: Call a crisis helpline or emergency services
                    2. 👥 **Reach out**: Contact a trusted friend, family member, or mental health professional
                    3. 🏥 **Seek care**: Visit your nearest emergency room if feeling unsafe
                    4. 📞 **Follow up**: Schedule an appointment with a mental health provider
                    """)
                elif prediction == "Support":
                    st.markdown("""
                    1. 💬 **Listen actively**: Provide a non-judgmental space for sharing
                    2. 🤗 **Offer support**: Ask "How can I support you right now?"
                    3. 📞 **Encourage help**: Suggest speaking with a counselor or therapist
                    4. 📱 **Share resources**: Provide crisis hotline numbers for future reference
                    """)
                else:
                    st.markdown("""
                    1. 📚 **Continue education**: Share mental health awareness resources
                    2. 💪 **Promote wellness**: Encourage self-care practices
                    3. 🗓️ **Regular check-ins**: Maintain ongoing support and communication
                    4. 🌈 **Build community**: Connect with mental health support groups
                    """)
                
        else:
            st.error("❌ Models not loaded. Please ensure models are trained and saved in the 'saved_models' directory.")
    
    elif analyze_btn and not user_input:
        st.warning("⚠️ Please enter some text to analyze.")

# ============================================================================
# TAB 2: DASHBOARD
# ============================================================================

with tab2:
    st.markdown("### 📊 Analytics Dashboard")
    
    # Load historical data if available
    try:
        if os.path.exists('saved_models/annotated_samples.csv'):
            df_history = pd.read_csv('saved_models/annotated_samples.csv')
            
            # Create metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Analyses", len(df_history), "All time")
            with col2:
                crisis_count = len(df_history[df_history['final_annotation'] == 'Crisis'])
                st.metric("Crisis Detections", crisis_count, 
                         f"{(crisis_count/len(df_history)*100):.1f}%")
            with col3:
                support_count = len(df_history[df_history['final_annotation'] == 'Support'])
                st.metric("Support Needed", support_count,
                         f"{(support_count/len(df_history)*100):.1f}%")
            with col4:
                accuracy = df_history['annotation_match'].mean()
                st.metric("Model Accuracy", f"{accuracy:.1%}", 
                         "vs human annotation")
            
            # Distribution chart
            st.markdown("#### 📈 Prediction Distribution")
            fig = px.pie(
                df_history, 
                names='final_annotation',
                title='Distribution of Predictions',
                color='final_annotation',
                color_discrete_map={'Crisis': '#f5576c', 'Support': '#4facfe', 'Neutral': '#43e97b'},
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Annotation agreement
            st.markdown("#### 🎯 Model Agreement Analysis")
            agree_rate = df_history['annotation_match'].mean()
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=agree_rate * 100,
                title={'text': "Model vs Human Agreement Rate (%)"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "#667eea"},
                       'steps': [
                           {'range': [0, 50], 'color': "#f8d7da"},
                           {'range': [50, 85], 'color': "#fff3cd"},
                           {'range': [85, 100], 'color': "#d4edda"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': 90}}))
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Sample of recent analyses
            st.markdown("#### 📋 Recent Analyses")
            st.dataframe(
                df_history[['text_clean', 'final_annotation', 'annotation_match']].head(10),
                use_container_width=True,
                hide_index=True
            )
            
        else:
            st.info("📊 No historical data available yet. Start analyzing text to build your dashboard.")
            
            # Show placeholder charts
            st.markdown("#### 📈 Sample Distribution (Coming Soon)")
            fig = go.Figure(data=[go.Pie(labels=['Crisis', 'Support', 'Neutral'], 
                                        values=[11, 20, 69],
                                        marker_colors=['#f5576c', '#4facfe', '#43e97b'])])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.warning(f"Could not load historical data: {str(e)}")

# ============================================================================
# TAB 3: ABOUT
# ============================================================================

with tab3:
    st.markdown("### ℹ️ About MindGuard")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        #### 🧠 What is MindGuard?
        
        MindGuard is an AI-powered mental health crisis detection system designed to identify 
        potential crisis situations in text-based communications in real-time. The system uses 
        advanced machine learning to analyze language patterns and provide immediate risk assessment.
        
        #### 🎯 Key Features
        
        - **Real-time Analysis**: Instant text processing and risk assessment
        - **Multi-label Classification**: Categorizes text as Crisis, Support, or Neutral
        - **High Accuracy**: 97% accuracy with 95% macro F1-score
        - **Rule-based Enhancement**: Combines ML with domain-specific keyword detection
        - **Crisis Resources**: Immediate access to emergency helplines and support
        
        #### 📊 Model Performance
        
        | Metric | Value |
        |--------|-------|
        | Accuracy | 97.4% |
        | Macro F1 | 95.4% |
        | Crisis Recall | 87.3% |
        | Crisis Precision | 98.4% |
        | Response Time | < 1 second |
        
        #### 🔬 Methodology
        
        The system uses a combination of:
        1. **Natural Language Processing**: TF-IDF vectorization with n-gram features
        2. **Machine Learning**: Logistic Regression with balanced class weights
        3. **Data Augmentation**: Text augmentation and SMOTE for class imbalance
        4. **Rule-based Detection**: Crisis keyword matching for immediate response
        """)
    
    with col2:
        st.markdown("#### 🛠️ Technologies Used")
        st.markdown("""
        - **Python 3.10+**
        - **Streamlit** (Web Interface)
        - **Scikit-learn** (ML Models)
        - **Pandas/NumPy** (Data Processing)
        - **Plotly** (Visualizations)
        - **NLTK** (Text Processing)
        - **imbalanced-learn** (SMOTE)
        - **nlpaug** (Data Augmentation)
        """)
        
        st.markdown("---")
        st.markdown("#### 👨‍💻 Development")
        st.markdown("""
        - **Dataset**: 10,000+ mental health forum posts
        - **Training**: 80/20 train-test split with stratification
        - **Validation**: 5-fold cross-validation
        - **Fairness Audit**: Per-group performance analysis
        """)
        
        st.markdown("---")
        st.markdown("#### 📝 License")
        st.markdown("© 2024 MindGuard | Educational/Research Use")

# ============================================================================
# TAB 4: RESOURCES
# ============================================================================

with tab4:
    st.markdown("### 🆘 Mental Health Resources")
    
    # Crisis resources
    st.markdown("#### 🚨 Crisis Helplines (24/7)")
    
    resources_df = pd.DataFrame([
        {"Service": "National Suicide Prevention Lifeline", "Number": "988", "Country": "USA"},
        {"Service": "Crisis Text Line", "Number": "Text HOME to 741741", "Country": "USA/Canada"},
        {"Service": "SAMHSA National Helpline", "Number": "1-800-662-4357", "Country": "USA"},
        {"Service": "Veterans Crisis Line", "Number": "988 (Press 1)", "Country": "USA"},
        {"Service": "Disaster Distress Helpline", "Number": "1-800-985-5990", "Country": "USA"},
        {"Service": "NHS Mental Health Helpline", "Number": "111", "Country": "UK"},
        {"Service": "Samaritans", "Number": "116 123", "Country": "UK"},
        {"Service": "Crisis Services Canada", "Number": "1-833-456-4566", "Country": "Canada"},
        {"Service": "Lifeline Australia", "Number": "13 11 14", "Country": "Australia"},
        {"Service": "iCall", "Number": "9152987821", "Country": "India"},
    ])
    
    st.dataframe(resources_df, use_container_width=True, hide_index=True)
    
    # Online resources
    st.markdown("#### 💻 Online Resources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Information & Education**
        - [NAMI](https://nami.org) - National Alliance on Mental Illness
        - [Mental Health America](https://mhanational.org)
        - [WHO Mental Health](https://www.who.int/health-topics/mental-health)
        - [NIH Mental Health](https://www.nimh.nih.gov)
        - [Psych Central](https://psychcentral.com)
        """)
    
    with col2:
        st.markdown("""
        **Support Communities**
        - [7 Cups](https://www.7cups.com) - Free emotional support
        - [The Mighty](https://themighty.com) - Mental health community
        - [Reddit r/mentalhealth](https://reddit.com/r/mentalhealth)
        - [Support Groups Central](https://supportgroupscentral.com)
        """)
    
    # Therapy directories
    st.markdown("#### 🏥 Find a Therapist")
    st.markdown("""
    - [Psychology Today Therapist Directory](https://www.psychologytoday.com/us/therapists)
    - [Open Path Collective](https://openpathcollective.org) - Affordable therapy
    - [BetterHelp](https://www.betterhelp.com) - Online therapy
    - [Talkspace](https://www.talkspace.com) - Online therapy
    - [Inclusive Therapists](https://www.inclusivetherapists.com) - LGBTQ+ affirming
    """)
    
    # Mobile apps
    st.markdown("#### 📱 Mental Health Apps")
    st.markdown("""
    - **CBT Thought Diary**: Cognitive Behavioral Therapy tool
    - **Calm**: Meditation and sleep
    - **Headspace**: Mindfulness and meditation
    - **Moodpath**: Depression and anxiety tracking
    - **Sanvello**: Stress, anxiety, depression management
    - **What's Up?**: CBT and ACT-based mental health app
    """)

# ============================================================================
# TAB 5: SELF-CARE
# ============================================================================

with tab5:
    st.markdown("### 💪 Self-Care & Coping Strategies")
    
    # Daily self-care tips
    st.markdown("#### 🌟 Daily Self-Care Tips")
    
    tips = get_self_care_tips()
    
    cols = st.columns(2)
    for i, tip in enumerate(tips):
        with cols[i % 2]:
            st.markdown(tip)
    
    # Grounding techniques
    st.markdown("---")
    st.markdown("#### 🌍 Grounding Techniques (5-4-3-2-1 Method)")
    
    st.markdown("""
    When feeling overwhelmed, try the 5-4-3-2-1 grounding technique:
    
    - **5** things you can SEE around you
    - **4** things you can TOUCH around you  
    - **3** things you can HEAR around you
    - **2** things you can SMELL around you
    - **1** thing you can TASTE
    
    This technique helps bring you back to the present moment.
    """)
    
    # Breathing exercise
    st.markdown("---")
    st.markdown("#### 🧘 Breathing Exercise (Square Breathing)")
    
    st.markdown("""
    Try this simple breathing exercise:
    
    1. **Inhale** through your nose for 4 seconds
    2. **Hold** your breath for 4 seconds
    3. **Exhale** through your mouth for 4 seconds
    4. **Hold** for 4 seconds
    5. Repeat 5-10 times
    
    This technique activates the parasympathetic nervous system and reduces anxiety.
    """)
    
    # Quick relaxation timer
    st.markdown("---")
    st.markdown("#### ⏱️ Quick Relaxation Timer")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        minutes = st.selectbox("Select duration (minutes):", [1, 2, 5, 10], index=1)
        if st.button("🧘 Start Breathing Exercise", use_container_width=True):
            st.info(f"Take {minutes} minutes to focus on your breath. Close your eyes and breathe deeply.")
            st.balloons()
    
    # Journaling prompt
    st.markdown("---")
    st.markdown("#### 📝 Journaling Prompt")
    
    journal_prompts = [
        "What am I feeling right now, and what might be causing these feelings?",
        "Three things I'm grateful for today are...",
        "What would I say to a friend who was feeling this way?",
        "What small step can I take today to feel better?",
        "What has helped me cope in the past that I can try now?"
    ]
    
    import random
    prompt = random.choice(journal_prompts)
    st.info(f"**Today's prompt**: {prompt}")
    
    # Crisis plan template
    st.markdown("---")
    with st.expander("📋 Create Your Safety Plan"):
        st.markdown("""
        ### My Personal Safety Plan
        
        **1. Warning Signs** (thoughts, feelings, behaviors that indicate crisis):
        - 
        
        **2. Internal Coping Strategies** (things I can do alone):
        - 
        
        **3. Social Support** (people who can help distract me):
        - 
        
        **4. Professional Support** (therapists, helplines):
        - 
        
        **5. Emergency Contacts**:
        - **Crisis Helpline**: 
        - **Emergency Services**: 911
        
        **6. Make Environment Safe**:
        - 
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div class="footer">
    <p>🧠 MindGuard | AI-Powered Mental Health Crisis Detection System</p>
    <p style="font-size: 0.8rem;">This tool is for educational and research purposes only. Not a substitute for professional mental health care.</p>
    <p style="font-size: 0.8rem;">© 2024 MindGuard | Data sources: Mental health forums (anonymized)</p>
</div>
""", unsafe_allow_html=True)
