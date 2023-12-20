# streamlit_app.py

# streamlit_app.py
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import streamlit as st
import pickle
import joblib
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained model and TF-IDF vectorizer
model = joblib.load("model_1.pkl")
tfidf_vectorizer = joblib.load("model_1_tfidf.pkl")

# Define the preprocess function
def preprocess(text):
    # Your preprocessing steps here
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Streamlit app layout
st.title("Sentiment Analysis App")

# Input box for the user to enter a review
user_input = st.text_area("Enter your review:")

# Preprocess the user input
cleaned_input = preprocess(user_input)
st.write("Cleaned Input:", cleaned_input)  # Debugging line

# Vectorize the input
input_tfidf = tfidf_vectorizer.transform([cleaned_input])
st.write("TF-IDF Vector:", input_tfidf)  # Debugging line

# Make prediction
prediction = model.predict(input_tfidf)[0]

# Display the result
st.subheader("Sentiment Prediction:")
st.write(prediction)

# Additional Streamlit components and visualizations can be added here
