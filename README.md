Here's your updated **README** with the **local Streamlit URL** added in a clean, professional way:

---

# Sentiment Analysis Web App (Streamlit)

An interactive web app built using Streamlit to analyze sentiment (Positive, Negative, Neutral) from user input text using a trained machine learning model.

## Features

* Enter a sentence or review and get real-time sentiment prediction
* View model confidence with a bar chart
* Simple UI using Streamlit
* Lightweight ML model using TF-IDF + Logistic Regression

## Tech Stack

* Python
* Scikit-learn
* Streamlit
* Pandas
* Joblib

## Model

Trained on a small sample dataset of labeled sentiment phrases using a pipeline:

* `TfidfVectorizer`
* `LogisticRegression`

## Run Locally

To run the app locally, use:

```bash
streamlit run app.py
```

Once running, access the app at:
**[http://localhost:8501](http://localhost:8501)**


