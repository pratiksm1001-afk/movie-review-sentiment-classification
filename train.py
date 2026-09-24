import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================================================
# 1. SETTINGS
# =========================================================

DATASET = "data/IMDB Dataset.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# =========================================================
# 2. LOAD DATA
# =========================================================

if not os.path.exists(DATASET):
    raise FileNotFoundError(
        f"{DATASET} not found. "
        "Make sure IMDB Dataset.csv is inside the data folder."
    )

df = pd.read_csv(DATASET)

print("=" * 60)
print("MOVIE REVIEW SENTIMENT CLASSIFICATION")
print("=" * 60)

print("\nDataset loaded successfully!")
print("Number of reviews:", len(df))


# =========================================================
# 3. DATA CLEANING
# =========================================================

df = df.dropna(
    subset=["review", "sentiment"]
)

df = df.drop_duplicates(
    subset=["review"]
)

df = df[
    df["sentiment"].isin(
        ["positive", "negative"]
    )
]

print("Reviews after cleaning:", len(df))


# =========================================================
# 4. TEXT PREPROCESSING
# =========================================================

def clean_text(text):

    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Keep only letters and spaces
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


df["clean_review"] = df["review"].apply(
    clean_text
)


# =========================================================
# 5. CONVERT SENTIMENT INTO NUMBERS
# =========================================================

df["label"] = df["sentiment"].map({
    "negative": 0,
    "positive": 1
})

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())


# =========================================================
# 6. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_review"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

print("\nTraining reviews:", len(X_train))
print("Testing reviews:", len(X_test))


# =========================================================
# 7. CREATE MACHINE LEARNING MODELS
# =========================================================

models = {

    "Naive Bayes": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words=None,
                ngram_range=(1, 2),
                max_features=50000,
                sublinear_tf=True
            )
        ),
        (
            "model",
            MultinomialNB(
                alpha=0.5
            )
        )
    ]),

    "Logistic Regression": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words=None,
                ngram_range=(1, 2),
                max_features=50000,
                sublinear_tf=True
            )
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]),

    "Linear SVM": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words=None,
                ngram_range=(1, 2),
                max_features=50000,
                sublinear_tf=True
            )
        ),
        (
            "model",
            LinearSVC()
        )
    ])
}


# =========================================================
# 8. TRAIN AND EVALUATE MODELS
# =========================================================

results = {}
trained_models = {}

for name, model in models.items():

    print("\n")
    print("=" * 60)
    print(name)
    print("=" * 60)

    print("Training model...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results[name] = accuracy

    trained_models[name] = model

    print(
        f"\nAccuracy: {accuracy * 100:.2f}%"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Negative",
                "Positive"
            ]
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("Confusion Matrix:")
    print(cm)


# =========================================================
# 9. MODEL COMPARISON
# =========================================================

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

for name, accuracy in results.items():

    print(
        f"{name}: "
        f"{accuracy * 100:.2f}%"
    )


# =========================================================
# 10. SAVE NAIVE BAYES MODEL
# =========================================================

naive_bayes_model = trained_models[
    "Naive Bayes"
]

model_path = (
    "models/naive_bayes_model.pkl"
)

joblib.dump(
    naive_bayes_model,
    model_path
)

print("\n")
print("=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(
    f"Naive Bayes model saved successfully!"
)

print(
    f"Location: {model_path}"
)


# =========================================================
# 11. TEST SAMPLE REVIEWS
# =========================================================

def predict_review(review):

    cleaned_review = clean_text(
        review
    )

    prediction = naive_bayes_model.predict(
        [cleaned_review]
    )[0]

    probability = (
        naive_bayes_model
        .predict_proba(
            [cleaned_review]
        )[0]
    )

    confidence = (
        max(probability) * 100
    )

    if prediction == 1:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return sentiment, confidence


# =========================================================
# 12. SAMPLE PREDICTIONS
# =========================================================

test_reviews = [

    "This movie was absolutely fantastic. I loved it!",

    "The movie was boring and a complete waste of time.",

    "Amazing acting and a wonderful story.",

    "I did not enjoy this movie at all."
]


print("\n")
print("=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)


for review in test_reviews:

    sentiment, confidence = (
        predict_review(review)
    )

    print("\nReview:")
    print(review)

    print(
        f"Prediction: {sentiment}"
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )


# =========================================================
# 13. COMPLETION MESSAGE
# =========================================================

print("\n")
print("=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(
    "\nYour Naive Bayes model is ready."
)

print(
    "Next step: create the Streamlit application."
)