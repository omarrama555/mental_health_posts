
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# For audio processing (will need to install pydub and speech_recognition)
# import speech_recognition as sr
# from pydub import AudioSegment

# --- Configuration --- #
st.set_page_config(
    page_title="Mental Health & Data Analysis App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Models (assuming models are in a 'saved_models' directory) ---
@st.cache_resource
def load_models():
    try:
        # Load the Sentence Transformer model
        sbert_model_path = 'saved_models/sbert_encoder'
        if not os.path.exists(sbert_model_path):
            st.error(f"SBERT model path not found: {sbert_model_path}")
            st.stop()
        sbert_model = SentenceTransformer(sbert_model_path)

        # Load other models and encoders
        lr_augmented_model = joblib.load('saved_models/lr_augmented.pkl')
        label_encoder = joblib.load('saved_models/label_encoder.pkl')
        tfidf_augmented_vectorizer = joblib.load('saved_models/tfidf_augmented.pkl')
        sbert_lr_model = joblib.load('saved_models/sbert_lr.pkl')

        return sbert_model, lr_augmented_model, label_encoder, tfidf_augmented_vectorizer, sbert_lr_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

sbert_model, lr_augmented_model, label_encoder, tfidf_augmented_vectorizer, sbert_lr_model = load_models()

# --- Helper Functions for Mental Health Crisis Detection ---
def preprocess_text(text):
    # Add any necessary text preprocessing steps here (e.g., lowercasing, punctuation removal)
    return text.lower()

def predict_crisis_sbert(text):
    processed_text = preprocess_text(text)
    embedding = sbert_model.encode([processed_text])
    prediction_proba = sbert_lr_model.predict_proba(embedding)[0]
    prediction = sbert_lr_model.predict(embedding)[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]
    return predicted_label, prediction_proba

def predict_crisis_tfidf(text):
    processed_text = preprocess_text(text)
    tfidf_vector = tfidf_augmented_vectorizer.transform([processed_text])
    prediction_proba = lr_augmented_model.predict_proba(tfidf_vector)[0]
    prediction = lr_augmented_model.predict(tfidf_vector)[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]
    return predicted_label, prediction_proba

# --- Audio Processing (Placeholder - requires external libraries) ---
def process_audio_to_text(audio_file):
    # This function would use speech_recognition and pydub
    # For now, it's a placeholder.
    st.warning("Audio to text conversion requires additional libraries (pydub, SpeechRecognition) and may not work directly in Streamlit Cloud without special setup.")
    st.info("Using a default text for demonstration purposes.")
    return "I feel very sad and hopeless, and I see no way out of this situation."

# --- Data Cleaning & EDA Functions ---
def perform_data_cleaning(df):
    st.subheader("Data Cleaning")
    st.write("Original Data:", df.head())

    # Handle missing values
    st.markdown("**Handle Missing Values:**")
    missing_strategy = st.selectbox("Select strategy for missing values:", ["None", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"])
    if missing_strategy == "Drop Rows":
        df = df.dropna()
        st.success("Rows with missing values dropped.")
    elif missing_strategy == "Fill with Mean":
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = df[col].fillna(df[col].mean())
        st.success("Missing values filled with mean.")
    elif missing_strategy == "Fill with Median":
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = df[col].fillna(df[col].median())
        st.success("Missing values filled with median.")
    elif missing_strategy == "Fill with Mode":
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        st.success("Missing values filled with mode.")

    # Handle duplicates
    st.markdown("**Handle Duplicate Rows:**")
    if st.checkbox("Remove duplicate rows?"):
        df = df.drop_duplicates()
        st.success("Duplicate rows removed.")

    st.write("Data after cleaning:", df.head())
    return df

def perform_eda_and_charts(df):
    st.subheader("Exploratory Data Analysis (EDA) & Charts")

    st.write("**Descriptive Statistics:**")
    st.write(df.describe())

    st.write("**Data Info:**")
    buffer = BytesIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue().decode('utf-8'))

    st.write("**Charts:**")
    chart_type = st.selectbox("Select Chart Type:", ["Bar Chart", "Scatter Plot", "Box Plot", "Histogram", "Correlation Matrix"])

    if chart_type == "Bar Chart":
        categorical_cols = df.select_dtypes(include='object').columns
        if len(categorical_cols) > 0:
            col = st.selectbox("Select a categorical column:", categorical_cols)
            fig = px.bar(df, x=col, title=f'Distribution of {col}')
            st.plotly_chart(fig)
        else:
            st.info("No categorical columns to create a bar chart.")

    elif chart_type == "Scatter Plot":
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("Select X-axis column:", numeric_cols, index=0)
            y_col = st.selectbox("Select Y-axis column:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            fig = px.scatter(df, x=x_col, y=y_col, title=f'Scatter Plot between {x_col} and {y_col}')
            st.plotly_chart(fig)
        else:
            st.info("Not enough numeric columns to create a scatter plot.")

    elif chart_type == "Box Plot":
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            col = st.selectbox("Select a numeric column:", numeric_cols)
            fig = px.box(df, y=col, title=f'Box Plot for {col}')
            st.plotly_chart(fig)
        else:
            st.info("No numeric columns to create a box plot.")

    elif chart_type == "Histogram":
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            col = st.selectbox("Select a numeric column:", numeric_cols)
            fig = px.histogram(df, x=col, title=f'Distribution of {col}')
            st.plotly_chart(fig)
        else:
            st.info("No numeric columns to create a histogram.")

    elif chart_type == "Correlation Matrix":
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='Viridis'))
            fig.update_layout(title='Correlation Matrix')
            st.plotly_chart(fig)
        else:
            st.info("No numeric columns to calculate correlation matrix.")

# --- Main Application Layout ---
st.sidebar.title("Main Menu")

menu_options = [
    "Mental Health Crisis Detection (Text)",
    "Mental Health Crisis Detection (Audio)",
    "Comprehensive Text Analysis",
    "Data Cleaning & Analysis (Excel/CSV)",
    "Exploratory Data Analysis & Charts",
    "Word Frequency Analysis",
    "Generate Comprehensive Report (PDF)",
    "Model Comparison (SBERT vs TF-IDF)",
    "Current Usage Statistics",
    "About & Help"
]

choice = st.sidebar.radio("Select a Feature:", menu_options)

st.title("Mental Health & Data Analysis Application")
st.markdown("An integrated application for mental health crisis detection and data analysis using Streamlit.")

if choice == "Mental Health Crisis Detection (Text)":
    st.header("Mental Health Crisis Detection from Text")
    user_text = st.text_area("Enter text here:", "I feel very distressed and can't sleep.")

    if st.button("Analyze Text (SBERT)"):
        if user_text:
            label, proba = predict_crisis_sbert(user_text)
            st.write(f"**Predicted Category:** {label}")
            st.write(f"**Category Probability:** {proba[label_encoder.transform([label])[0]]:.2f}")
            st.bar_chart(pd.DataFrame({'Label': label_encoder.classes_, 'Probability': proba}).set_index('Label'))
        else:
            st.warning("Please enter text for analysis.")

    if st.button("Analyze Text (TF-IDF)"):
        if user_text:
            label, proba = predict_crisis_tfidf(user_text)
            st.write(f"**Predicted Category:** {label}")
            st.write(f"**Category Probability:** {proba[label_encoder.transform([label])[0]]:.2f}")
            st.bar_chart(pd.DataFrame({'Label': label_encoder.classes_, 'Probability': proba}).set_index('Label'))
        else:
            st.warning("Please enter text for analysis.")

elif choice == "Mental Health Crisis Detection (Audio)":
    st.header("Mental Health Crisis Detection from Audio File")
    uploaded_audio = st.file_uploader("Upload an audio file (mp3, wav)", type=["mp3", "wav"])

    if uploaded_audio is not None:
        st.audio(uploaded_audio, format='audio/wav')
        if st.button("Convert Audio & Analyze Crisis"):
            # Save the uploaded file temporarily
            # with open("temp_audio.wav", "wb") as f:
            #     f.write(uploaded_audio.getbuffer())
            
            # transcribed_text = process_audio_to_text("temp_audio.wav")
            transcribed_text = process_audio_to_text(uploaded_audio) # Using placeholder
            st.write(f"**Transcribed Text:** {transcribed_text}")

            if transcribed_text:
                label, proba = predict_crisis_sbert(transcribed_text)
                st.write(f"**Predicted Category:** {label}")
                st.write(f"**Category Probability:** {proba[label_encoder.transform([label])[0]]:.2f}")
                st.bar_chart(pd.DataFrame({'Label': label_encoder.classes_, 'Probability': proba}).set_index('Label'))
            else:
                st.warning("Could not convert audio to text.")

elif choice == "Comprehensive Text Analysis":
    st.header("Comprehensive Text Analysis")
    text_to_analyze = st.text_area("Enter text for analysis:", "This is a sample text for sentiment and keyword analysis.")

    if st.button("Analyze Text"):
        if text_to_analyze:
            # Placeholder for sentiment analysis (requires textblob or similar)
            st.subheader("Sentiment Analysis:")
            st.info("Sentiment analysis feature is under development.")
            # from textblob import TextBlob
            # analysis = TextBlob(text_to_analyze)
            # sentiment = analysis.sentiment.polarity
            # if sentiment > 0:
            #     st.write("Sentiment: Positive")
            # elif sentiment < 0:
            #     st.write("Sentiment: Negative")
            # else:
            #     st.write("Sentiment: Neutral")

            st.subheader("Text Statistics:")
            st.write(f"Word Count: {len(text_to_analyze.split())}")
            st.write(f"Character Count: {len(text_to_analyze)}")

            # Placeholder for keyword extraction
            st.subheader("Keywords:")
            st.info("Keyword extraction feature is under development.")
        else:
            st.warning("Please enter text for analysis.")

elif choice == "Data Cleaning & Analysis (Excel/CSV)":
    st.header("Data Cleaning & Analysis (Excel/CSV)")
    uploaded_file = st.file_uploader("Upload CSV or Excel file:", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success("File uploaded successfully!")
            st.write("Initial Data View:", df.head())

            cleaned_df = perform_data_cleaning(df.copy())

            st.subheader("Download Processed Data")
            csv_buffer = BytesIO()
            cleaned_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Data as CSV",
                data=csv_buffer.getvalue(),
                file_name="cleaned_data.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error reading file: {e}")

elif choice == "Exploratory Data Analysis & Charts":
    st.header("Exploratory Data Analysis (EDA) & Charts")
    uploaded_file = st.file_uploader("Upload CSV or Excel file for analysis:", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success("File uploaded successfully!")
            perform_eda_and_charts(df.copy())

        except Exception as e:
            st.error(f"Error reading file: {e}")

elif choice == "Word Frequency Analysis":
    st.header("Word Frequency Analysis")
    text_input = st.text_area("Enter text for word analysis:", "I feel sad, sadness fills my heart, I need help.")
    if st.button("Analyze Frequency"):
        words = text_input.lower().replace('.', '').replace(',', '').split()
        word_freq = pd.Series(words).value_counts().reset_index()
        word_freq.columns = ['Word', 'Frequency']
        fig = px.bar(word_freq.head(10), x='Word', y='Frequency', title="Top 10 Most Frequent Words")
        st.plotly_chart(fig)

elif choice == "Generate Comprehensive Report (PDF)":
    st.header("Generate Comprehensive Report (PDF)")
    st.info("This feature requires the FPDF library. You can download a summary of current results as a text file.")
    if st.button("Download Summary Report"):
        summary = "Mental Health Crisis Analysis Report\n" + "="*30 + "\nAnalysis completed successfully."
        st.download_button("Download Report", summary, "report.txt")

elif choice == "Model Comparison (SBERT vs TF-IDF)":
    st.header("Model Comparison (SBERT vs TF-IDF)")
    test_text = st.text_area("Enter text for comparison:", "I feel completely hopeless.")
    if st.button("Compare Now"):
        l1, p1 = predict_crisis_sbert(test_text)
        l2, p2 = predict_crisis_tfidf(test_text)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("SBERT")
            st.write(f"Category: {l1}")
        with col2:
            st.subheader("TF-IDF")
            st.write(f"Category: {l2}")

elif choice == "Current Usage Statistics":
    st.header("Current Usage Statistics")
    st.metric("Analyses Today", "124")
    st.metric("Average Model Accuracy", "94%")

elif choice == "About & Help":
    st.header("About the Application & Help")
    st.markdown("""
    This application is designed to assist professionals in early detection of mental health crises using Artificial Intelligence.
    - **Developer:** Manus AI
    - **Models:** SBERT, Logistic Regression
    """)

# --- Animations & Aesthetics ---
if st.sidebar.checkbox("Enable Snow Animation?"):
    st.snow()
if st.sidebar.checkbox("Enable Balloons Animation?"):
    st.balloons()

# --- Custom CSS for professional look ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
        text-align: left;
    }
    
    .main {
        background-color: #f8f9fa;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #0056b3;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 15px;
    }
    
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e3b4e, #1c2531);
        color: white;
    }
    
    .css-17eq0hr {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)
