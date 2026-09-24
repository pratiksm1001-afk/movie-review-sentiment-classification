import streamlit as st
import joblib
import re
import pandas as pd

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Movie Review Classifier",
    page_icon="🎬",
    layout="centered"
)

# =========================================================
# DARK CINEMATIC CSS
# =========================================================

st.markdown("""
<style>

/* ================= MAIN APP ================= */

.stApp {
    background:
        radial-gradient(
            circle at top right,
            #312e81 0%,
            transparent 35%
        ),
        radial-gradient(
            circle at bottom left,
            #0f766e 0%,
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #020617 0%,
            #0f172a 50%,
            #111827 100%
        );

    color: #f8fafc;
}

/* Remove light container background */

.main {
    background: transparent !important;
}

.block-container {
    background: transparent !important;
    padding-top: 2rem;
}

/* ================= TEXT ================= */

p {
    color: #e2e8f0 !important;
}

label {
    color: #f8fafc !important;
}

h1, h2, h3 {
    color: #ffffff !important;
}

/* ================= TITLE ================= */

h1 {
    text-align: center;
    font-size: 42px !important;
    font-weight: 800 !important;

    background: linear-gradient(
        90deg,
        #38bdf8,
        #818cf8,
        #c084fc
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ================= SUBTITLE ================= */

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1 !important;
    margin-bottom: 25px;
}

/* ================= TEXT AREA ================= */

textarea {
    border-radius: 14px !important;

    border: 1px solid #475569 !important;

    background-color: #0f172a !important;

    color: #ffffff !important;

    font-size: 16px !important;
}

textarea::placeholder {
    color: #94a3b8 !important;
}

/* ================= BUTTON ================= */

.stButton > button {

    width: 100%;

    border-radius: 12px;

    height: 50px;

    font-size: 17px;

    font-weight: 700;

    color: white;

    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    );

    border: none;

    transition: 0.3s;
}

.stButton > button:hover {

    transform: scale(1.02);

    box-shadow:
        0 0 20px rgba(124, 58, 237, 0.5);
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #020617 0%,
            #111827 100%
        );

    border-right: 1px solid #334155;
}

section[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

/* ================= RESULT BOX ================= */

.result-box {

    padding: 20px;

    border-radius: 16px;

    text-align: center;

    margin-top: 15px;

    margin-bottom: 15px;

    font-size: 24px;

    font-weight: 800;

    background: rgba(
        30,
        41,
        59,
        0.8
    );

    border: 1px solid #475569;

    box-shadow:
        0 8px 25px rgba(0,0,0,0.3);
}

/* ================= METRICS ================= */

[data-testid="stMetric"] {

    background: rgba(
        30,
        41,
        59,
        0.85
    );

    border: 1px solid #475569;

    border-radius: 14px;

    padding: 15px;

    box-shadow:
        0 5px 20px rgba(0,0,0,0.25);
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* ================= SELECT BOX ================= */

div[data-baseweb="select"] > div {

    background-color: #0f172a !important;

    border: 1px solid #475569 !important;

    border-radius: 12px !important;

    color: white !important;
}

/* ================= EXPANDER ================= */

.streamlit-expanderHeader {

    background: #1e293b !important;

    color: white !important;

    border-radius: 10px;
}

/* ================= INFO BOX ================= */

div[data-testid="stAlert"] {

    border-radius: 12px;
}

/* ================= DATAFRAME ================= */

[data-testid="stDataFrame"] {

    background: #0f172a !important;

    border-radius: 12px;

}

/* ================= CODE BOX ================= */

.stCodeBlock {

    background: #020617 !important;

    border: 1px solid #334155;

    border-radius: 12px;
}

/* ================= FOOTER ================= */

.footer {

    text-align: center;

    color: #94a3b8;

    font-size: 14px;

    margin-top: 25px;

    padding: 15px;
}

/* ================= DIVIDER ================= */

hr {

    border-color: #334155 !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = "models/naive_bayes_model.pkl"


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = joblib.load(MODEL_PATH)

except FileNotFoundError:

    st.error(
        "❌ Model not found. Please run train.py first."
    )

    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================================================
# PREDICTION
# =========================================================

def predict_review(review):

    cleaned_review = clean_text(review)

    prediction = model.predict(
        [cleaned_review]
    )[0]

    probability = model.predict_proba(
        [cleaned_review]
    )[0]

    negative_probability = probability[0] * 100

    positive_probability = probability[1] * 100

    if prediction == 1:

        sentiment = "Positive"

    else:

        sentiment = "Negative"

    return (
        sentiment,
        negative_probability,
        positive_probability
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🎬 Project Information")

    st.write("### Movie Review Classification")

    st.write(
        "A Machine Learning application that "
        "classifies movie reviews as Positive "
        "or Negative."
    )

    st.write("### 🔧 Technologies")

    st.write("• Python")
    st.write("• Pandas")
    st.write("• Scikit-learn")
    st.write("• TF-IDF")
    st.write("• Multinomial Naive Bayes")
    st.write("• Streamlit")

    st.write("### 📊 Dataset")

    st.write(
        "IMDB Movie Reviews Dataset"
    )

    st.write("### 🤖 Model")

    st.write(
        "TF-IDF + Multinomial Naive Bayes"
    )

    st.write("### 🎯 Output")

    st.write(
        "Positive 😊 / Negative 😞"
    )


# =========================================================
# MAIN TITLE
# =========================================================

st.markdown(
    '<h1>🎬 Movie Review Sentiment Classifier</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered movie review sentiment analysis'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# REVIEW INPUT
# =========================================================

review = st.text_area(
    "✍️ Enter your movie review:",
    height=150,
    placeholder=(
        "Example: This movie was amazing "
        "and I really enjoyed it!"
    )
)


# =========================================================
# ANALYZE REVIEW
# =========================================================

if st.button("🔍 Analyze Review"):

    if review.strip() == "":

        st.warning(
            "⚠️ Please enter a movie review."
        )

    else:

        (
            sentiment,
            negative_probability,
            positive_probability
        ) = predict_review(review)

        # Save history

        st.session_state.prediction_history.append({

            "Review": review,

            "Sentiment": sentiment,

            "Positive": (
                f"{positive_probability:.2f}%"
            ),

            "Negative": (
                f"{negative_probability:.2f}%"
            )
        })

        st.subheader(
            "📊 Prediction Result"
        )

        if sentiment == "Positive":

            st.markdown(
                '<div class="result-box">'
                '😊 POSITIVE REVIEW'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="result-box">'
                '😞 NEGATIVE REVIEW'
                '</div>',
                unsafe_allow_html=True
            )

        st.write(
            "### 📈 Sentiment Probabilities"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "😞 Negative",
                f"{negative_probability:.2f}%"
            )

        with col2:

            st.metric(
                "😊 Positive",
                f"{positive_probability:.2f}%"
            )

        st.write(
            "**Negative Probability**"
        )

        st.progress(
            int(negative_probability)
        )

        st.write(
            "**Positive Probability**"
        )

        st.progress(
            int(positive_probability)
        )

        st.info(
            "These percentages are probabilities "
            "estimated by the trained Naive Bayes model."
        )


# =========================================================
# SAMPLE REVIEWS
# =========================================================

st.divider()

st.subheader(
    "💡 Try a Sample Review"
)

sample = st.selectbox(
    "Choose a sample:",
    [
        "Select a review",

        "This movie was absolutely fantastic. "
        "I loved it!",

        "The movie was boring and a complete "
        "waste of time.",

        "Amazing acting and a wonderful story.",

        "I did not enjoy this movie at all."
    ]
)


# =========================================================
# ANALYZE SAMPLE
# =========================================================

if sample != "Select a review":

    if st.button("🔍 Analyze Sample"):

        (
            sentiment,
            negative_probability,
            positive_probability
        ) = predict_review(sample)

        # Save history

        st.session_state.prediction_history.append({

            "Review": sample,

            "Sentiment": sentiment,

            "Positive": (
                f"{positive_probability:.2f}%"
            ),

            "Negative": (
                f"{negative_probability:.2f}%"
            )
        })

        st.write(
            "**Review:**",
            sample
        )

        if sentiment == "Positive":

            st.success(
                "😊 Prediction: Positive"
            )

        else:

            st.error(
                "😞 Prediction: Negative"
            )

        st.write(
            "### 📈 Sample Probabilities"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "😞 Negative",
                f"{negative_probability:.2f}%"
            )

        with col2:

            st.metric(
                "😊 Positive",
                f"{positive_probability:.2f}%"
            )

        st.write(
            "**Negative Probability**"
        )

        st.progress(
            int(negative_probability)
        )

        st.write(
            "**Positive Probability**"
        )

        st.progress(
            int(positive_probability)
        )


# =========================================================
# PREDICTION HISTORY
# =========================================================

if st.session_state.prediction_history:

    st.divider()

    st.subheader(
        "📋 Prediction History"
    )

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button(
        "🗑️ Clear Prediction History"
    ):

        st.session_state.prediction_history = []

        st.rerun()


# =========================================================
# HOW SYSTEM WORKS
# =========================================================

st.divider()

st.subheader(
    "⚙️ How This System Works"
)

st.write(
    "The system follows these Machine Learning steps:"
)

step1, step2 = st.columns(2)

with step1:

    st.info(
        "### 1️⃣ Review Input\n"
        "The user enters a movie review."
    )

    st.info(
        "### 2️⃣ Text Cleaning\n"
        "The review is converted to lowercase "
        "and unnecessary characters are removed."
    )

    st.info(
        "### 3️⃣ TF-IDF\n"
        "The cleaned text is converted into "
        "numerical features."
    )

with step2:

    st.info(
        "### 4️⃣ Naive Bayes\n"
        "The trained Multinomial Naive Bayes "
        "model analyzes the features."
    )

    st.info(
        "### 5️⃣ Prediction\n"
        "The model predicts Positive or Negative."
    )

    st.info(
        "### 6️⃣ Probability\n"
        "The estimated probability for both "
        "classes is displayed."
    )


# =========================================================
# ABOUT PROJECT
# =========================================================

st.divider()

st.subheader(
    "📚 About the Project"
)

with st.expander(
    "🔎 What is Sentiment Analysis?"
):

    st.write(
        "Sentiment analysis is a Natural Language "
        "Processing technique used to identify "
        "the sentiment expressed in text."
    )

    st.write(
        "This project classifies movie reviews "
        "as Positive or Negative."
    )


with st.expander(
    "🤖 Why Naive Bayes?"
):

    st.write(
        "Multinomial Naive Bayes is commonly used "
        "for text classification."
    )

    st.write(
        "It uses probability to classify text "
        "into different categories."
    )


with st.expander(
    "🔢 What is TF-IDF?"
):

    st.write(
        "TF-IDF stands for Term Frequency-Inverse "
        "Document Frequency."
    )

    st.write(
        "It converts text into numerical features "
        "based on word importance."
    )


with st.expander(
    "📊 What is the IMDB Dataset?"
):

    st.write(
        "The project uses the IMDB movie review "
        "dataset containing reviews labeled "
        "as Positive or Negative."
    )


# =========================================================
# MACHINE LEARNING PIPELINE
# =========================================================

st.subheader(
    "🔄 Machine Learning Pipeline"
)

st.code(
    """
Movie Review
     ↓
Text Cleaning
     ↓
TF-IDF Vectorization
     ↓
Multinomial Naive Bayes
     ↓
Sentiment Prediction
     ↓
Positive / Negative
     ↓
Probability
""",
    language="text"
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    '<div class="footer">'
    '🎬 <b>Movie Review Classification</b><br>'
    'TF-IDF + Multinomial Naive Bayes<br>'
    'Diploma CSE Mini Project'
    '</div>',
    unsafe_allow_html=True
)