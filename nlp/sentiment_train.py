import json, random, pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Demo dataset (positive/negative/neutral)
samples = [
    ("маш сайн үйлчилгээ байна", "pos"), ("таалагдсан", "pos"),
    ("дажгүй", "neu"), ("зүгээр", "neu"),
    ("муу байна", "neg"), ("таалагдсангүй", "neg")
]

X, y = zip(*samples)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000))
])
pipe.fit(X_train, y_train)
print("✅ sentiment_model.pkl saved")
joblib.dump(pipe, "nlp/sentiment_model.pkl")
