import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib


df = pd.read_csv('sentiment_data.csv')

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

model.fit(df['text'], df['label'])
joblib.dump(model, 'sentiment_model.pkl')
