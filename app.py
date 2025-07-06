import streamlit as st
import joblib
import numpy as np

model = joblib.load('sentiment_model.pkl')
labels = ['negative', 'neutral', 'positive']

st.title("🧠 Sentiment Analysis Web App")
st.write("Enter a sentence to predict sentiment:")

user_input = st.text_area("💬 Input Text")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([user_input])[0]
        proba = model.predict_proba([user_input])[0]

        st.success(f"🔍 Predicted Sentiment: **{prediction.capitalize()}**")

        st.subheader("📊 Sentiment Probabilities")
        st.bar_chart({
            'Sentiment': labels,
            'Probability': proba
        })
