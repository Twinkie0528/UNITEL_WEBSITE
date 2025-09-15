import os, re, json, pickle, numpy as np, random
from urllib.parse import urlencode
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize

lemmatizer = WordNetLemmatizer()
model   = load_model("chatbot_model.h5")
intents = json.loads(open("job_intents.json", encoding="utf-8").read())
words   = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

def clean_up_sentence(sentence):
    return [lemmatizer.lemmatize(w.lower()) for w in wordpunct_tokenize(sentence)]

def bow(sentence, vocab):
    sw = clean_up_sentence(sentence)
    return np.array([1 if w in sw else 0 for w in vocab])

def predict_class(sentence):
    res = model.predict(np.array([bow(sentence, words)]), verbose=0)[0]
    th = 0.4
    ranked = [[i, r] for i, r in enumerate(res) if r > th]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[i], "p": float(p)} for i, p in ranked]

def get_resp(tag):
    for it in intents["intents"]:
        if it["tag"] == tag:
            return random.choice(it["responses"])
    return "🤖 …"

def parse_days(text, default=7):
    m = re.search(r'(\d+)\s*(хоног|days?)', text, re.I)
    return max(1, int(m.group(1))) if m else default

def handle_intent(tag, msg):
    if tag == "fb_report":
        q = urlencode({"days": 7})
        return f"📊 Graph API тайлан:\n- CSV: /report/fb.csv?{q}\n- XLSX: /report/fb.xlsx?{q}\n(⚠️ Token тохируулсан үед бодит дата)"
    if tag == "sentiment_report":
        d = parse_days(msg, 7)
        q = urlencode({"days": d})
        return f"🧠 Сүүлийн {d} хоногийн sentiment тайлан:\n- CSV: /report/sentiment.csv?{q}\n- XLSX: /report/sentiment.xlsx?{q}"
    if tag == "ads_report":
        q = urlencode({"days": 30})
        return f"📡 Ads тайлан:\n- CSV: /report/ads.csv?{q}\n- XLSX: /report/ads.xlsx?{q}\nДэлгэрэнгүй: /admin/ads"
    if tag == "help":
        return get_resp("help")
    return None

def chatbot_response(msg):
    ranked = predict_class(msg)
    if ranked:
        tag = ranked[0]["intent"]
        custom = handle_intent(tag, msg)
        return custom or get_resp(tag)
    return get_resp("fallback")
