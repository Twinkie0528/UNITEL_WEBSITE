import joblib
model = joblib.load("nlp/sentiment_model.pkl")
def predict_sentiment(text: str) -> str:
    return model.predict([text])[0]
