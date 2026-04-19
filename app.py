# app.py
import streamlit as st
import joblib
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

# -----------------------------
# Step 1: Load model & vectorizer
# -----------------------------
MODEL_PATH = "models/logistic_model.pkl"
VECT_PATH = "models/tfidf_vectorizer.pkl"

# Check if models exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
    st.error("Model or vectorizer not found! Run main.py first to train and save them.")
    st.stop()

# Load saved model & vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

# -----------------------------
# Step 2: Streamlit UI
# -----------------------------
st.title("Sentiment Analysis Dashboard")
st.write("Enter text below to predict sentiment:")

user_input = st.text_area("Paste text here:", height=150)

if st.button("Analyze") and user_input.strip():
    # -----------------------------
    # Step 3: Preprocess input
    # -----------------------------
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

    def clean_text_fast(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # remove URLs
        text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation/numbers
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]
        return " ".join(tokens)

    cleaned_input = clean_text_fast(user_input)

    # -----------------------------
    # Step 4: Predict sentiment
    # -----------------------------
    X_input = vectorizer.transform([cleaned_input])
    pred = model.predict(X_input)[0]
    pred_prob = model.predict_proba(X_input)[0]

    # Map model classes dynamically
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    st.write(f"**Predicted Sentiment:** {sentiment_map[pred]}")

    # Show confidence scores dynamically
    st.write("**Confidence Scores:**")
    for i, cls in enumerate(model.classes_):
        st.write(f"{sentiment_map[cls]}: {pred_prob[i]*100:.2f}%")

    # -----------------------------
    # Step 5: Generate word cloud
    # -----------------------------
    st.write("**Word Cloud of Input Text:**")
    wordcloud = WordCloud(width=600, height=300, background_color='white').generate(cleaned_input)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    