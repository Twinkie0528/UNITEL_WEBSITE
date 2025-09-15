import os, pandas as pd, streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/")
db = MongoClient(MONGO_URL).chatbot_db

st.set_page_config(page_title="Unitel Dash", page_icon="🟢", layout="wide")
st.title("Unitel — Internal Dashboard")

tab1, tab2 = st.tabs(["📡 Ads", "🧠 Sentiment"])

with tab1:
    days = st.slider("Сүүлийн (хоног)", 7, 90, 30)
    pipeline = [
        {"$sort": {"detected_at": 1}},
        {"$group": {
            "_id": "$ad_id",
            "site": {"$last": "$site"},
            "brand": {"$last": "$brand"},
            "placement": {"$last": "$placement"},
            "url": {"$last": "$url"},
            "first_seen": {"$first": "$detected_at"},
            "last_seen": {"$last": "$detected_at"},
            "status": {"$last": "$status"}
        }}
    ]
    rows = list(db.ads_events.aggregate(pipeline))
    df = pd.DataFrame(rows).rename(columns={"_id":"ad_id"})
    st.dataframe(df, use_container_width=True)
    if not df.empty and "site" in df:
        st.bar_chart(df["site"].value_counts())

with tab2:
    df = pd.DataFrame(list(db.fb_comments.find({}, {"_id":0})))
    st.dataframe(df, use_container_width=True)
    st.info("Sentiment model-аа холбосны дараа график нэмнэ.")
