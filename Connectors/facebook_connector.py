import os
from datetime import datetime, timedelta
import requests

FB_TOKEN  = os.getenv("FB_ACCESS_TOKEN")
FB_PAGEID = os.getenv("FB_PAGE_ID")
BASE      = "https://graph.facebook.com/v19.0"

def has_token():
    return bool(FB_TOKEN and FB_PAGEID)

def fetch_recent_posts(limit=10, since_hours=24):
    if not has_token(): return []
    # TODO: бодит дуудлагаа хийх (token бэлэн болмогц)
    return []

def fetch_comments(post_id, limit=200):
    if not has_token(): return []
    return []
