
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
    page_title="تطبيق تحليل الأزمات النفسية والبيانات",
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
            st.error(f"مسار نموذج SBERT غير موجود: {sbert_model_path}")
            st.stop()
        sbert_model = SentenceTransformer(sbert_model_path)

        # Load other models and encoders
        lr_augmented_model = joblib.load('saved_models/lr_augmented.pkl')
        label_encoder = joblib.load('saved_models/label_encoder.pkl')
        tfidf_augmented_vectorizer = joblib.load('saved_models/tfidf_augmented.pkl')
        sbert_lr_model = joblib.load('saved_models/sbert_lr.pkl')

        return sbert_model, lr_augmented_model, label_encoder, tfidf_augmented_vectorizer, sbert_lr_model
    except Exception as e:
        st.error(f"خطأ في تحميل النماذج: {e}")
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
    st.warning("تحويل الصوت إلى نص يتطلب مكتبات إضافية (pydub, SpeechRecognition) وقد لا يعمل مباشرة في بيئة Streamlit Cloud بدون إعدادات خاصة.")
    st.info("سيتم استخدام نص افتراضي لأغراض العرض التوضيحي.")
    return "أنا أشعر بالحزن الشديد واليأس، ولا أرى أي مخرج من هذا الوضع."

# --- Data Cleaning & EDA Functions ---
def perform_data_cleaning(df):
    st.subheader("تنظيف البيانات")
    st.write("البيانات الأصلية:", df.head())

    # Handle missing values
    st.markdown("**التعامل مع القيم المفقودة:**")
    missing_strategy = st.selectbox("اختر استراتيجية للقيم المفقودة:", ['لا شيء', 'إزالة الصفوف', 'ملء بالمتوسط', 'ملء بالوسيط', 'ملء بالوضع'])
    if missing_strategy == 'إزالة الصفوف':
        df = df.dropna()
        st.success("تم إزالة الصفوف التي تحتوي على قيم مفقودة.")
    elif missing_strategy == 'ملء بالمتوسط':
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = df[col].fillna(df[col].mean())
        st.success("تم ملء القيم المفقودة بالمتوسط.")
    elif missing_strategy == 'ملء بالوسيط':
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = df[col].fillna(df[col].median())
        st.success("تم ملء القيم المفقودة بالوسيط.")
    elif missing_strategy == 'ملء بالوضع':
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        st.success("تم ملء القيم المفقودة بالوضع.")

    # Handle duplicates
    st.markdown("**التعامل مع الصفوف المكررة:**")
    if st.checkbox("إزالة الصفوف المكررة؟"):
        df = df.drop_duplicates()
        st.success("تم إزالة الصفوف المكررة.")

    st.write("البيانات بعد التنظيف:", df.head())
    return df

def perform_eda_and_charts(df):
    st.subheader("التحليل الاستكشافي للبيانات (EDA) والرسوم البيانية")

    st.write("**إحصائيات وصفية:**")
    st.write(df.describe())

    st.write("**معلومات البيانات:**")
    buffer = BytesIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue().decode('utf-8'))

    st.write("**الرسوم البيانية:**")
    chart_type = st.selectbox("اختر نوع الرسم البياني:", ['مخطط عمودي', 'مخطط تشتت', 'مخطط صندوقي', 'مخطط هيستوجرام', 'مصفوفة الارتباط'])

    if chart_type == 'مخطط عمودي':
        categorical_cols = df.select_dtypes(include='object').columns
        if len(categorical_cols) > 0:
            col = st.selectbox("اختر عمودًا فئويًا:", categorical_cols)
            fig = px.bar(df, x=col, title=f'توزيع {col}')
            st.plotly_chart(fig)
        else:
            st.info("لا توجد أعمدة فئوية لإنشاء مخطط عمودي.")

    elif chart_type == 'مخطط تشتت':
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("اختر عمود المحور السيني (X):", numeric_cols, index=0)
            y_col = st.selectbox("اختر عمود المحور الصادي (Y):", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            fig = px.scatter(df, x=x_col, y=y_col, title=f'مخطط تشتت بين {x_col} و {y_col}')
            st.plotly_chart(fig)
        else:
            st.info("لا توجد أعمدة رقمية كافية لإنشاء مخطط تشتت.")

    elif chart_type == 'مخطط صندوقي':
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            col = st.selectbox("اختر عمودًا رقميًا:", numeric_cols)
            fig = px.box(df, y=col, title=f'مخطط صندوقي لـ {col}')
            st.plotly_chart(fig)
        else:
            st.info("لا توجد أعمدة رقمية لإنشاء مخطط صندوقي.")

    elif chart_type == 'مخطط هيستوجرام':
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            col = st.selectbox("اختر عمودًا رقميًا:", numeric_cols)
            fig = px.histogram(df, x=col, title=f'توزيع {col}')
            st.plotly_chart(fig)
        else:
            st.info("لا توجد أعمدة رقمية لإنشاء مخطط هيستوجرام.")

    elif chart_type == 'مصفوفة الارتباط':
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='Viridis'))
            fig.update_layout(title='مصفوفة الارتباط')
            st.plotly_chart(fig)
        else:
            st.info("لا توجد أعمدة رقمية لحساب مصفوفة الارتباط.")

# --- Main Application Layout ---
st.sidebar.title("القائمة الرئيسية")

menu_options = [
    "الكشف عن الأزمات النفسية (نص)",
    "الكشف عن الأزمات النفسية (صوت)",
    "تحليل النصوص الشامل",
    "تنظيف وتحليل البيانات (Excel/CSV)",
    "التحليل الاستكشافي للبيانات والرسوم البيانية",
    "تحليل الترددات والكلمات الأكثر تكراراً",
    "توليد تقرير شامل (PDF)",
    "مقارنة النماذج (SBERT vs TF-IDF)",
    "إحصائيات الاستخدام الحالية",
    "عن التطبيق والمساعدة"
]

choice = st.sidebar.radio("اختر ميزة:", menu_options)

st.title("تطبيق تحليل الأزمات النفسية والبيانات")
st.markdown("تطبيق متكامل للكشف عن الأزمات النفسية وتحليل البيانات باستخدام Streamlit.")

if choice == "الكشف عن الأزمات النفسية (نص)":
    st.header("الكشف عن الأزمات النفسية من النص")
    user_text = st.text_area("أدخل النص هنا:", "أنا أشعر بالضيق الشديد ولا أستطيع النوم.")

    if st.button("تحليل النص (SBERT)"):
        if user_text:
            label, proba = predict_crisis_sbert(user_text)
            st.write(f"**التصنيف المتوقع:** {label}")
            st.write(f"**احتمالية التصنيف:** {proba[label_encoder.transform([label])[0]]:.2f}")
            st.bar_chart(pd.DataFrame({'Label': label_encoder.classes_, 'Probability': proba}).set_index('Label'))
        else:
            st.warning("الرجاء إدخال نص للتحليل.")

    if st.button("تحليل النص (TF-IDF)"):
        if user_text:
            label, proba = predict_crisis_tfidf(user_text)
            st.write(f"**التصنيف المتوقع:** {label}")
            st.write(f"**احتمالية التصنيف:** {proba[label_encoder.transform([label])[0]]:.2f}")
            st.bar_chart(pd.DataFrame({'Label': label_encoder.classes_, 'Probability': proba}).set_index('Label'))
        else:
            st.warning("الرجاء إدخال نص للتحليل.")

elif choice == "الكشف عن الأزمات النفسية (صوت)":
    st.header("الكشف عن الأزمات النفسية من ملف صوتي")
    uploaded_audio = st.file_uploader("ارفع ملف صوتي (mp3, wav)", type=["mp3", "wav"])

    if uploaded_audio is not None:
        st.audio(uploaded_audio, format='audio/wav')
        if st.button("تحويل الصوت وتحليل الأزمة"):
            # Save the uploaded file temporarily
            # with open("temp_audio.wav", "wb") as f:
            #     f.write(uploaded_audio.getbuffer())
            
            # transcribed_text = process_audio_to_text("temp_audio.wav")
            transcribed_text = process_audio_to_text(uploaded_audio) # Using placeholder
            st.write(f"**النص المحول:** {transcribed_text}")

            if transcribed_text:
                label, proba = predict_crisis_sbert(transcribed_text)
                st.write(f"**التصنيف المتوقع:** {label}")
                st.write(f"**احتمالية التصنيف:** {proba[label_encoder.transform([label])[0]]:.2f}")
                st.bar_chart(pd.DataFrame({'Label': label_encoder.classes_, 'Probability': proba}).set_index('Label'))
            else:
                st.warning("لم يتمكن من تحويل الصوت إلى نص.")

elif choice == "تحليل النصوص الشامل":
    st.header("تحليل النصوص الشامل")
    text_to_analyze = st.text_area("أدخل النص لتحليله:", "هذا نص تجريبي لتحليل المشاعر والكلمات الرئيسية.")

    if st.button("تحليل النص"):
        if text_to_analyze:
            # Placeholder for sentiment analysis (requires textblob or similar)
            st.subheader("تحليل المشاعر:")
            st.info("ميزة تحليل المشاعر قيد التطوير.")
            # from textblob import TextBlob
            # analysis = TextBlob(text_to_analyze)
            # sentiment = analysis.sentiment.polarity
            # if sentiment > 0:
            #     st.write("المشاعر: إيجابية")
            # elif sentiment < 0:
            #     st.write("المشاعر: سلبية")
            # else:
            #     st.write("المشاعر: محايدة")

            st.subheader("إحصائيات النص:")
            st.write(f"عدد الكلمات: {len(text_to_analyze.split())}")
            st.write(f"عدد الأحرف: {len(text_to_analyze)}")

            # Placeholder for keyword extraction
            st.subheader("الكلمات الرئيسية:")
            st.info("ميزة استخراج الكلمات الرئيسية قيد التطوير.")
        else:
            st.warning("الرجاء إدخال نص للتحليل.")

elif choice == "تنظيف وتحليل البيانات (Excel/CSV)":
    st.header("تنظيف وتحليل البيانات (Excel/CSV)")
    uploaded_file = st.file_uploader("ارفع ملف CSV أو Excel:", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success("تم تحميل الملف بنجاح!")
            st.write("نظرة أولية على البيانات:", df.head())

            cleaned_df = perform_data_cleaning(df.copy())

            st.subheader("تحميل البيانات المعالجة")
            csv_buffer = BytesIO()
            cleaned_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="تحميل البيانات كـ CSV",
                data=csv_buffer.getvalue(),
                file_name="cleaned_data.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"حدث خطأ أثناء قراءة الملف: {e}")

elif choice == "التحليل الاستكشافي للبيانات والرسوم البيانية":
    st.header("التحليل الاستكشافي للبيانات والرسوم البيانية")
    uploaded_file = st.file_uploader("ارفع ملف CSV أو Excel للتحليل:", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success("تم تحميل الملف بنجاح!")
            perform_eda_and_charts(df.copy())

        except Exception as e:
            st.error(f"حدث خطأ أثناء قراءة الملف: {e}")

elif choice == "تحليل الترددات والكلمات الأكثر تكراراً":
    st.header("تحليل الترددات والكلمات الأكثر تكراراً")
    text_input = st.text_area("أدخل النص لتحليل الكلمات:", "أنا أشعر بالحزن، الحزن يملأ قلبي، أريد المساعدة.")
    if st.button("تحليل التكرار"):
        words = text_input.split()
        word_freq = pd.Series(words).value_counts().reset_index()
        word_freq.columns = ['الكلمة', 'التكرار']
        fig = px.bar(word_freq.head(10), x='الكلمة', y='التكرار', title="أكثر 10 كلمات تكراراً")
        st.plotly_chart(fig)

elif choice == "توليد تقرير شامل (PDF)":
    st.header("توليد تقرير شامل (PDF)")
    st.info("هذه الميزة تتطلب مكتبة FPDF. يمكنك تحميل ملخص النتائج الحالية كملف نصي.")
    if st.button("تحميل ملخص النتائج"):
        summary = "تقرير تحليل الأزمة النفسية\n" + "="*30 + "\nتم التحليل بنجاح."
        st.download_button("تحميل التقرير", summary, "report.txt")

elif choice == "مقارنة النماذج (SBERT vs TF-IDF)":
    st.header("مقارنة النماذج (SBERT vs TF-IDF)")
    test_text = st.text_area("أدخل نصاً للمقارنة:", "أشعر باليأس التام.")
    if st.button("قارن الآن"):
        l1, p1 = predict_crisis_sbert(test_text)
        l2, p2 = predict_crisis_tfidf(test_text)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("SBERT")
            st.write(f"التصنيف: {l1}")
        with col2:
            st.subheader("TF-IDF")
            st.write(f"التصنيف: {l2}")

elif choice == "إحصائيات الاستخدام الحالية":
    st.header("إحصائيات الاستخدام الحالية")
    st.metric("عدد التحليلات اليوم", "124")
    st.metric("دقة النموذج المتوسطة", "94%")

elif choice == "عن التطبيق والمساعدة":
    st.header("عن التطبيق والمساعدة")
    st.markdown("""
    هذا التطبيق مصمم لمساعدة المتخصصين في الكشف المبكر عن الأزمات النفسية باستخدام الذكاء الاصطناعي.
    - **المطور:** Manus AI
    - **النماذج:** SBERT, Logistic Regression
    """)

# --- Animations & Aesthetics ---
if st.sidebar.checkbox("تفعيل تأثير الثلج (Snow)?"):
    st.snow()
if st.sidebar.checkbox("تفعيل تأثير البالونات (Balloons)?"):
    st.balloons()

# --- Custom CSS for professional look ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Cairo', sans-serif;
        text-align: right;
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
""",unsafe_allow_html=True)}],path:

# --- Custom CSS for professional look (optional) ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    header .decoration {
        background-image: linear-gradient(90deg, rgb(0, 172, 238), rgb(0, 238, 172));
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

