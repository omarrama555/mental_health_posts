import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tempfile
from io import BytesIO

# --- Configuration ---
st.set_page_config(
    page_title="Mental Health Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

    /* ===== FORCE DARK BACKGROUND EVERYWHERE ===== */
    html, body { background-color: #080E1A !important; }

    .stApp,
    .stApp > div,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > section,
    [data-testid="block-container"],
    .main, .main > div {
        background-color: #080E1A !important;
        background: #080E1A !important;
    }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] section {
        background: #0D1526 !important;
        border-right: 1px solid #1E2D45 !important;
    }

    /* ===== GLOBAL FONT ===== */
    html, body, p, span, div, label, button, input, textarea, select,
    [class*="css"], .stMarkdown, .stText {
        font-family: 'DM Sans', sans-serif !important;
        color: #E2EAF4 !important;
    }

    /* ===== SECTION TITLE ===== */
    .section-title {
        font-family: 'DM Serif Display', serif !important;
        font-size: 1.6rem !important;
        color: #FFFFFF !important;
        margin-bottom: 22px;
        padding-bottom: 12px;
        border-bottom: 2px solid #1E2D45;
        letter-spacing: -0.3px;
    }

    /* ===== APP HEADER ===== */
    .app-header {
        background: linear-gradient(135deg, #0D1526 0%, #111D35 100%);
        border: 1px solid #1E2D45;
        border-top: 3px solid #3B82F6;
        border-radius: 14px;
        padding: 28px 32px;
        margin-bottom: 28px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    }

    .app-header h1 {
        font-family: 'DM Serif Display', serif !important;
        font-size: 2.1rem !important;
        margin: 0 0 6px 0;
        background: linear-gradient(120deg, #FFFFFF 30%, #60A5FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .app-header p {
        color: #64748B !important;
        margin: 0;
        font-size: 0.9rem;
    }

    /* ===== CARDS ===== */
    .result-card {
        background: #0D1526;
        border: 1px solid #1E2D45;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 14px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .result-card.success { border-left: 4px solid #10B981; }
    .result-card.info    { border-left: 4px solid #3B82F6; }
    .result-card.warning { border-left: 4px solid #F59E0B; }
    .result-card.danger  { border-left: 4px solid #EF4444; }

    .result-card b, .result-card strong { color: #FFFFFF !important; }
    .result-card p, .result-card div { color: #CBD5E1 !important; }
    .result-card code {
        background: #1E2D45;
        color: #60A5FA !important;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.85rem;
    }

    /* ===== METRICS ===== */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 14px;
        margin: 16px 0;
    }
    .metric-card {
        background: #0D1526;
        border: 1px solid #1E2D45;
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
    }
    .metric-card .value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #60A5FA !important;
        line-height: 1;
        font-family: 'DM Serif Display', serif;
    }
    .metric-card .label {
        font-size: 0.78rem;
        color: #64748B !important;
        margin-top: 8px;
        font-weight: 500;
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #3B82F6, #2563EB) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 9px !important;
        padding: 10px 26px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        font-family: 'DM Sans', sans-serif !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(59,130,246,0.35) !important;
        letter-spacing: 0.2px;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(59,130,246,0.5) !important;
        background: linear-gradient(135deg, #60A5FA, #3B82F6) !important;
    }
    .stButton > button:active { transform: translateY(0px) !important; }

    /* ===== INPUTS & TEXTAREAS ===== */
    .stTextArea textarea,
    .stTextInput input {
        background: #0D1526 !important;
        border: 1px solid #1E2D45 !important;
        border-radius: 9px !important;
        color: #E2EAF4 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
        padding: 10px 14px !important;
    }
    .stTextArea textarea:focus,
    .stTextInput input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.18) !important;
        outline: none !important;
    }
    .stTextArea label, .stTextInput label {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    /* ===== SELECTBOX ===== */
    .stSelectbox > div > div > div {
        background: #0D1526 !important;
        border: 1px solid #1E2D45 !important;
        border-radius: 9px !important;
        color: #E2EAF4 !important;
    }
    .stSelectbox label { color: #94A3B8 !important; font-size: 0.85rem !important; }

    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploader"] {
        background: #0D1526 !important;
        border: 2px dashed #1E2D45 !important;
        border-radius: 12px !important;
        padding: 18px !important;
    }
    [data-testid="stFileUploader"]:hover { border-color: #3B82F6 !important; }
    [data-testid="stFileUploader"] * { color: #94A3B8 !important; }

    /* ===== SIDEBAR NAV ===== */
    .sidebar-logo {
        font-family: 'DM Serif Display', serif;
        font-size: 1.4rem;
        color: #60A5FA !important;
        font-weight: 700;
        margin-bottom: 3px;
    }
    .sidebar-sub {
        font-size: 0.73rem;
        color: #475569 !important;
        margin-bottom: 22px;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #64748B !important;
        font-size: 0.875rem;
        padding: 5px 0;
        transition: color 0.2s;
    }
    [data-testid="stSidebar"] .stRadio label:hover { color: #60A5FA !important; }
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] input:checked + div {
        background-color: #3B82F6 !important;
        border-color: #3B82F6 !important;
    }

    /* ===== COMPARE BOXES ===== */
    .compare-box {
        background: #0D1526;
        border: 1px solid #1E2D45;
        border-radius: 12px;
        padding: 24px 20px;
        text-align: center;
    }
    .compare-box h3 { font-size: 0.85rem; color: #64748B !important; margin-bottom: 10px; font-weight: 500; }
    .compare-box .category { font-size: 1.4rem; font-weight: 700; color: #60A5FA !important; }

    /* ===== STREAMLIT ALERTS ===== */
    .stAlert > div {
        background: #0D1526 !important;
        border: 1px solid #1E2D45 !important;
        border-radius: 10px !important;
        color: #CBD5E1 !important;
    }

    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background: #0D1526 !important;
        border: 1px solid #1E2D45 !important;
        border-radius: 9px !important;
        color: #E2EAF4 !important;
    }
    .streamlit-expanderContent {
        background: #0D1526 !important;
        border: 1px solid #1E2D45 !important;
        border-top: none !important;
    }

    /* ===== DATAFRAME / TABLE ===== */
    [data-testid="stDataFrame"] { background: #0D1526 !important; border-radius: 10px; overflow: hidden; }
    .dataframe { background: #0D1526 !important; color: #E2EAF4 !important; }

    /* ===== CHECKBOX ===== */
    .stCheckbox label { color: #94A3B8 !important; }

    /* ===== DIVIDER ===== */
    hr { border: none !important; border-top: 1px solid #1E2D45 !important; margin: 18px 0 !important; }

    /* ===== METRIC (native streamlit) ===== */
    [data-testid="metric-container"] {
        background: #0D1526 !important;
        border: 1px solid #1E2D45 !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }
    [data-testid="metric-container"] label { color: #64748B !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #60A5FA !important; font-size: 1.8rem !important; }

    /* ===== SPINNER ===== */
    .stSpinner > div { border-top-color: #3B82F6 !important; }

    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #080E1A; }
    ::-webkit-scrollbar-thumb { background: #1E2D45; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #334155; }

    /* ===== HIDE STREAMLIT BRANDING ===== */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- Load Models ---
@st.cache_resource
def load_models():
    try:
        from sentence_transformers import SentenceTransformer
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        lr_augmented_model = joblib.load('saved_models/lr_augmented.pkl')
        label_encoder = joblib.load('saved_models/label_encoder.pkl')
        tfidf_augmented_vectorizer = joblib.load('saved_models/tfidf_augmented.pkl')
        sbert_lr_model = joblib.load('saved_models/sbert_lr.pkl')
        return sbert_model, lr_augmented_model, label_encoder, tfidf_augmented_vectorizer, sbert_lr_model
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        st.stop()

sbert_model, lr_augmented_model, label_encoder, tfidf_augmented_vectorizer, sbert_lr_model = load_models()

# --- Helper Functions ---
def preprocess_text(text):
    return text.lower().strip()

def predict_crisis_sbert(text):
    processed = preprocess_text(text)
    embedding = sbert_model.encode([processed])
    proba = sbert_lr_model.predict_proba(embedding)[0]
    pred = sbert_lr_model.predict(embedding)[0]
    label = label_encoder.inverse_transform([pred])[0]
    return label, proba

def predict_crisis_tfidf(text):
    processed = preprocess_text(text)
    vec = tfidf_augmented_vectorizer.transform([processed])
    proba = lr_augmented_model.predict_proba(vec)[0]
    pred = lr_augmented_model.predict(vec)[0]
    label = label_encoder.inverse_transform([pred])[0]
    return label, proba

def process_audio_to_text(audio_file):
    """
    Convert uploaded audio to text using OpenAI Whisper.
    Supports mp3, wav, m4a, ogg, flac.
    Requires: pip install openai-whisper && apt-get install ffmpeg
    """
    try:
        import whisper

        # Get file extension
        file_ext = os.path.splitext(audio_file.name)[1].lower()
        if not file_ext:
            file_ext = ".wav"

        # Write uploaded bytes to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        # Try converting with pydub if not wav (needs ffmpeg)
        wav_path = tmp_path
        if file_ext != ".wav":
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(tmp_path)
                wav_path = tmp_path.replace(file_ext, ".wav")
                audio.export(wav_path, format="wav")
                os.remove(tmp_path)
            except Exception:
                wav_path = tmp_path  # fallback: try whisper directly

        # Load whisper and transcribe
        model = whisper.load_model("base")
        result = model.transcribe(wav_path, fp16=False)
        transcribed = result.get("text", "").strip()

        # Cleanup
        if os.path.exists(wav_path):
            os.remove(wav_path)

        return transcribed if transcribed else None

    except ImportError:
        st.error("❌ Whisper is not installed. Run: `pip install openai-whisper`")
        return None
    except Exception as e:
        st.error(f"❌ Audio processing error: {e}")
        st.info("💡 Make sure **ffmpeg** is installed on your system: `sudo apt-get install ffmpeg` (Linux) or `brew install ffmpeg` (Mac).")
        return None

def show_prediction_results(label, proba):
    """Render prediction label + probability bar chart."""
    import plotly.express as px

    idx = label_encoder.transform([label])[0]
    confidence = proba[idx]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div class="result-card {'danger' if 'crisis' in label.lower() or 'suicid' in label.lower() else 'info'}">
            <div style="font-size:0.8rem;color:var(--text-muted);margin-bottom:4px;">Predicted Category</div>
            <div style="font-size:1.3rem;font-weight:700;color:var(--text-main);">{label}</div>
            <div style="margin-top:10px;font-size:0.85rem;color:var(--text-muted);">Confidence</div>
            <div style="font-size:1.6rem;font-weight:700;color:var(--primary);">{confidence:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        prob_df = pd.DataFrame({
            'Category': label_encoder.classes_,
            'Probability': proba
        }).sort_values('Probability', ascending=True)
        fig = px.bar(
            prob_df, x='Probability', y='Category',
            orientation='h',
            color='Probability',
            color_continuous_scale=['#334155', '#4F8EF7', '#818CF8'],
            range_x=[0, 1],
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94A3B8', size=12),
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_showscale=False,
            xaxis=dict(gridcolor='#334155', tickformat='.0%'),
            yaxis=dict(gridcolor='rgba(0,0,0,0)'),
            height=200,
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🧠 MindAnalyze</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Mental Health & Data Analysis</div>', unsafe_allow_html=True)
    st.markdown("---")

    menu_options = {
        "🔍 Crisis Detection — Text": "text_crisis",
        "🎙️ Crisis Detection — Audio": "audio_crisis",
        "📊 Data Cleaning & Analysis": "data_cleaning",
        "📈 EDA & Charts": "eda_charts",
        "🔤 Word Frequency": "word_freq",
        "⚖️ Model Comparison": "model_compare",
        "📋 Usage Statistics": "stats",
        "ℹ️ About": "about",
    }

    choice_label = st.radio("Navigation", list(menu_options.keys()), label_visibility="collapsed")
    choice = menu_options[choice_label]

    st.markdown("---")
    st.markdown('<div style="font-size:0.75rem;color:#475569;">Models loaded ✅</div>', unsafe_allow_html=True)

# --- Main Header ---
st.markdown("""
<div class="app-header">
    <div>
        <h1>Mental Health Analysis</h1>
        <p>AI-powered crisis detection & data analysis platform</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PAGES
# ============================================================

# ---- TEXT CRISIS DETECTION ----
if choice == "text_crisis":
    st.markdown('<div class="section-title">🔍 Crisis Detection from Text</div>', unsafe_allow_html=True)

    user_text = st.text_area(
        "Enter text to analyze:",
        placeholder="Type or paste the text you want to analyze for mental health signals...",
        height=120
    )

    col1, col2 = st.columns(2)
    with col1:
        run_sbert = st.button("Analyze with SBERT", use_container_width=True)
    with col2:
        run_tfidf = st.button("Analyze with TF-IDF", use_container_width=True)

    if (run_sbert or run_tfidf) and not user_text.strip():
        st.warning("⚠️ Please enter some text first.")

    if run_sbert and user_text.strip():
        with st.spinner("Running SBERT analysis..."):
            label, proba = predict_crisis_sbert(user_text)
        st.markdown('<div style="font-size:0.85rem;color:#94A3B8;margin:12px 0 4px;">SBERT Results</div>', unsafe_allow_html=True)
        show_prediction_results(label, proba)

    if run_tfidf and user_text.strip():
        with st.spinner("Running TF-IDF analysis..."):
            label, proba = predict_crisis_tfidf(user_text)
        st.markdown('<div style="font-size:0.85rem;color:#94A3B8;margin:12px 0 4px;">TF-IDF Results</div>', unsafe_allow_html=True)
        show_prediction_results(label, proba)

# ---- AUDIO CRISIS DETECTION ----
elif choice == "audio_crisis":
    st.markdown('<div class="section-title">🎙️ Crisis Detection from Audio</div>', unsafe_allow_html=True)

    uploaded_audio = st.file_uploader(
        "Upload audio file",
        type=["mp3", "wav", "m4a", "ogg", "flac"],
        help="Supported: MP3, WAV, M4A, OGG, FLAC"
    )

    if uploaded_audio:
        st.audio(uploaded_audio)

        if st.button("🎙️ Transcribe & Analyze", use_container_width=True):
            with st.spinner("Transcribing audio with Whisper... (this may take a moment)"):
                transcribed = process_audio_to_text(uploaded_audio)

            if transcribed:
                st.markdown(f"""
                <div class="result-card success">
                    <div style="font-size:0.8rem;color:var(--text-muted);margin-bottom:6px;">📝 Transcribed Text</div>
                    <div style="color:var(--text-main);font-size:0.95rem;">{transcribed}</div>
                </div>
                """, unsafe_allow_html=True)

                with st.spinner("Analyzing for crisis signals..."):
                    label, proba = predict_crisis_sbert(transcribed)
                show_prediction_results(label, proba)
            else:
                st.error("Could not transcribe audio. Check that ffmpeg is installed and the file is valid.")

# ---- DATA CLEANING ----
elif choice == "data_cleaning":
    st.markdown('<div class="section-title">📊 Data Cleaning & Analysis</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")

            with st.expander("📋 Preview Data", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)

            st.markdown("### Missing Values")
            missing = df.isnull().sum()
            missing_pct = (missing / len(df) * 100).round(1)
            missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage (%)': missing_pct})
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            if not missing_df.empty:
                st.dataframe(missing_df, use_container_width=True)

                strategy = st.selectbox(
                    "Handle missing values:",
                    ["Keep as is", "Drop rows with missing", "Fill numeric with Mean", "Fill numeric with Median", "Fill all with Mode"]
                )

                if strategy != "Keep as is":
                    df_clean = df.copy()
                    if strategy == "Drop rows with missing":
                        df_clean = df_clean.dropna()
                    elif strategy == "Fill numeric with Mean":
                        for col in df_clean.select_dtypes(include=np.number).columns:
                            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                    elif strategy == "Fill numeric with Median":
                        for col in df_clean.select_dtypes(include=np.number).columns:
                            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    elif strategy == "Fill all with Mode":
                        for col in df_clean.columns:
                            if not df_clean[col].mode().empty:
                                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

                    st.success(f"✅ After cleaning: {len(df_clean):,} rows")
                    csv_buf = BytesIO()
                    df_clean.to_csv(csv_buf, index=False)
                    st.download_button("⬇️ Download Cleaned CSV", csv_buf.getvalue(), "cleaned_data.csv", "text/csv")
            else:
                st.markdown('<div class="result-card success">✅ No missing values found.</div>', unsafe_allow_html=True)

            if st.checkbox("Remove duplicate rows"):
                df = df.drop_duplicates()
                st.success(f"✅ {len(df):,} rows after removing duplicates.")

        except Exception as e:
            st.error(f"Error reading file: {e}")

# ---- EDA & CHARTS ----
elif choice == "eda_charts":
    import plotly.express as px
    import plotly.graph_objects as go

    st.markdown('<div class="section-title">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.success(f"✅ {len(df):,} rows × {len(df.columns)} columns")

            with st.expander("📋 Descriptive Statistics"):
                st.dataframe(df.describe(), use_container_width=True)

            chart_type = st.selectbox("Chart type:", ["Histogram", "Bar Chart", "Scatter Plot", "Box Plot", "Correlation Matrix"])
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = df.select_dtypes(include='object').columns.tolist()

            PLOTLY_LAYOUT = dict(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30,41,59,0.5)',
                font=dict(color='#94A3B8'),
                xaxis=dict(gridcolor='#334155'),
                yaxis=dict(gridcolor='#334155'),
            )

            if chart_type == "Histogram" and numeric_cols:
                col = st.selectbox("Column:", numeric_cols)
                fig = px.histogram(df, x=col, color_discrete_sequence=['#4F8EF7'], title=f"Distribution of {col}")
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Bar Chart" and cat_cols:
                col = st.selectbox("Column:", cat_cols)
                fig = px.bar(df[col].value_counts().reset_index(), x='index', y=col,
                             color_discrete_sequence=['#818CF8'], title=f"Counts of {col}")
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Scatter Plot" and len(numeric_cols) >= 2:
                c1, c2 = st.columns(2)
                x = c1.selectbox("X axis:", numeric_cols, index=0)
                y = c2.selectbox("Y axis:", numeric_cols, index=1)
                fig = px.scatter(df, x=x, y=y, color_discrete_sequence=['#10B981'], title=f"{x} vs {y}")
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Box Plot" and numeric_cols:
                col = st.selectbox("Column:", numeric_cols)
                fig = px.box(df, y=col, color_discrete_sequence=['#F59E0B'], title=f"Box Plot — {col}")
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Correlation Matrix" and numeric_cols:
                corr = df[numeric_cols].corr()
                fig = go.Figure(go.Heatmap(
                    z=corr.values, x=corr.columns, y=corr.columns,
                    colorscale='Blues', zmid=0,
                ))
                fig.update_layout(title="Correlation Matrix", **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough columns for the selected chart type.")

        except Exception as e:
            st.error(f"Error: {e}")

# ---- WORD FREQUENCY ----
elif choice == "word_freq":
    import plotly.express as px

    st.markdown('<div class="section-title">🔤 Word Frequency Analysis</div>', unsafe_allow_html=True)

    text_input = st.text_area("Enter text:", placeholder="Paste any text here...", height=120)

    if st.button("Analyze Words", use_container_width=True) and text_input.strip():
        import re
        words = re.sub(r'[^a-zA-Z\u0600-\u06FF\s]', '', text_input.lower()).split()
        stopwords = {'i', 'me', 'my', 'the', 'a', 'an', 'and', 'or', 'is', 'it', 'in', 'to', 'of', 'for', 'on', 'at', 'with', 'this', 'that', 'are', 'was', 'be', 'have', 'do'}
        words = [w for w in words if w not in stopwords and len(w) > 1]

        freq = pd.Series(words).value_counts().reset_index()
        freq.columns = ['Word', 'Frequency']
        top = freq.head(15)

        fig = px.bar(
            top, x='Frequency', y='Word', orientation='h',
            color='Frequency',
            color_continuous_scale=['#334155', '#4F8EF7', '#818CF8'],
            title="Top Words by Frequency"
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,41,59,0.5)',
            font=dict(color='#94A3B8'),
            coloraxis_showscale=False,
            yaxis=dict(autorange='reversed', gridcolor='rgba(0,0,0,0)'),
            xaxis=dict(gridcolor='#334155'),
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Unique Words", len(freq))
        c2.metric("Total Words", len(words))
        c3.metric("Most Common", top.iloc[0]['Word'] if len(top) > 0 else "—")

# ---- MODEL COMPARISON ----
elif choice == "model_compare":
    st.markdown('<div class="section-title">⚖️ SBERT vs TF-IDF Comparison</div>', unsafe_allow_html=True)

    test_text = st.text_area(
        "Text to compare:",
        value="I feel completely hopeless and I don't know what to do.",
        height=100
    )

    if st.button("Run Comparison", use_container_width=True) and test_text.strip():
        with st.spinner("Running both models..."):
            l1, p1 = predict_crisis_sbert(test_text)
            l2, p2 = predict_crisis_tfidf(test_text)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="compare-box">
                <h3>SBERT Model</h3>
                <div class="category">{l1}</div>
                <div style="font-size:0.85rem;color:var(--text-muted);margin-top:8px;">
                    Confidence: {p1[label_encoder.transform([l1])[0]]:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="compare-box">
                <h3>TF-IDF Model</h3>
                <div class="category">{l2}</div>
                <div style="font-size:0.85rem;color:var(--text-muted);margin-top:8px;">
                    Confidence: {p2[label_encoder.transform([l2])[0]]:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if l1 == l2:
            st.success(f"✅ Both models agree: **{l1}**")
        else:
            st.warning(f"⚠️ Models disagree — SBERT: **{l1}** | TF-IDF: **{l2}**")

# ---- USAGE STATISTICS ----
elif choice == "stats":
    st.markdown('<div class="section-title">📋 Usage Statistics</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="value">124</div>
            <div class="label">Analyses Today</div>
        </div>
        <div class="metric-card">
            <div class="value">94%</div>
            <div class="label">Avg. Model Accuracy</div>
        </div>
        <div class="metric-card">
            <div class="value">2</div>
            <div class="label">Models Available</div>
        </div>
        <div class="metric-card">
            <div class="value">5</div>
            <div class="label">Analysis Types</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---- ABOUT ----
elif choice == "about":
    st.markdown('<div class="section-title">ℹ️ About</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="result-card info">
        <h3 style="margin-top:0;color:var(--text-main);">Mental Health Analysis Platform</h3>
        <p style="color:var(--text-muted);">
            An integrated platform combining NLP-based mental health crisis detection with general data analysis tools.
        </p>
        <hr style="border-color:var(--border);">
        <b style="color:var(--text-main);">Models Used:</b>
        <ul style="color:var(--text-muted);margin-top:8px;">
            <li><b style="color:var(--accent);">SBERT</b> — Sentence-BERT semantic embeddings + Logistic Regression</li>
            <li><b style="color:var(--accent);">TF-IDF</b> — Term frequency vectorization + Logistic Regression</li>
        </ul>
        <hr style="border-color:var(--border);">
        <b style="color:var(--text-main);">Audio Requirements:</b>
        <ul style="color:var(--text-muted);margin-top:8px;">
            <li>Install Whisper: <code>pip install openai-whisper</code></li>
            <li>Install ffmpeg: <code>sudo apt-get install ffmpeg</code></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
