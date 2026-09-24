# Movie Review Sentiment Classification Using Naive Bayes

## Project Description
This is a Diploma CSE mini project that uses Machine Learning and Natural Language Processing (NLP) to classify movie reviews as **Positive** or **Negative**.

## Objective
The main objective is to automatically identify the sentiment of a movie review using the **Multinomial Naive Bayes** algorithm.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF
- Multinomial Naive Bayes
- Joblib
- Streamlit

## Dataset
The project uses the **IMDB Movie Reviews Dataset** containing movie reviews and their sentiment labels.

## Methodology
1. Load the movie review dataset.
2. Clean and preprocess the review text.
3. Convert text into numerical features using **TF-IDF**.
4. Train the **Multinomial Naive Bayes** model.
5. Evaluate the model.
6. Save the trained model using Joblib.
7. Use Streamlit to create a web application for prediction.

## Project Workflow

Movie Review
↓
Text Cleaning
↓
TF-IDF
↓
Naive Bayes
↓
Sentiment Prediction
↓
Positive / Negative

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt