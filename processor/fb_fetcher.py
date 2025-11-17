# ======================================================
# fb_fetcher.py — Unified Facebook/Instagram/Ads Data Fetcher (Date Range Ready)
# Author: Unitel AI Hub (2025 edition, downloads moved to /data/downloads)
# ======================================================

from __future__ import annotations
import os, requests, csv, time, json, pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

# -------------------- CONFIG --------------------
GRAPH = "https://graph.facebook.com/v20.0"
PAGE_ID = os.getenv("FB_PAGE_ID")
IG_ID = os.getenv("IG_BUSINESS_ID")
ACCESS_TOKEN = os.getenv("FB_ACCESS_TOKEN")
AD_ACCOUNT_ID = os.getenv("FB_AD_ACCOUNT_ID")

MAX_RETRY = 3
RETRY_DELAY = 1.5
DEFAULT_TIMEOUT = 25

# -------------------- DIRECTORIES (ШИНЭЧИЛСЭН ХЭСЭГ) --------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOWNLOADS_DIR = DATA_DIR / "downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)  # ensure folder exists


# ======================================================
# BASIC UTILITIES
# ======================================================
def _get(url: str, params: Optional[Dict[str, Any]] = None, retries: int = MAX_RETRY) -> Dict[str, Any]:
    """GET request with retry, pagination & backoff."""
    params = params or {}
    params["access_token"] = ACCESS_TOKEN
    result_data = []
    next_url = url

    for attempt in range(retries):
        try:
            r = requests.get(next_url, params=params, timeout=DEFAULT_TIMEOUT)
            if r.ok:
                data = r.json()
                if "data" in data:
                    result_data.extend(data["data"])
                    next_url = data.get("paging", {}).get("next")
                    if not next_url:
                        break
                    params = {}  # next page already has token
                else:
                    return data
            else:
                print(f"[Retry {attempt+1}] HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(RETRY_DELAY * (attempt + 1))
        except Exception as e:
            print(f"⚠️ Graph request error: {e}")
            time.sleep(RETRY_DELAY * (attempt + 1))

    return {"data": result_data}


def _to_df(items: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert list of dicts to normalized DataFrame."""
    if not items:
        return pd.DataFrame()
    df = pd.json_normalize(items)
    df.columns = [c.replace(".", "_") for c in df.columns]
    return df


# ======================================================
# PAGE INSIGHTS
# ======================================================
def get_page_insights(fields=None) -> Dict[str, Any]:
    """Facebook Page insights (followers, engagement, reach)."""
    url = f"{GRAPH}/{PAGE_ID}"
    params = {"fields": fields or "followers_count,fan_count,link,name"}
    return _get(url, params)


# ======================================================
# PAGE POSTS
# ======================================================
def list_posts(since: Optional[str] = None, until: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """List page posts (supports since/until date filter)."""
    params = {
        "fields": "id,message,created_time,permalink_url,reactions.summary(true),comments.summary(true)",
        "limit": limit
    }
    if since: params["since"] = since
    if until: params["until"] = until

    data = []
    url = f"{GRAPH}/{PAGE_ID}/posts"

    while True:
        res = _get(url, params)
        chunk = res.get("data", [])
        data.extend(chunk)
        next_url = res.get("paging", {}).get("next")
        if not next_url:
            break
        url = next_url
        params = {}

    print(f"✅ {len(data)} posts fetched for page {PAGE_ID}")
    return data


# ======================================================
# COMMENTS (with since/until)
# ======================================================
def list_comments(post_id: str, limit: int = 200,
                  since: Optional[str] = None, until: Optional[str] = None) -> List[Dict[str, Any]]:
    """List comments for a given post (supports date filter)."""
    url = f"{GRAPH}/{post_id}/comments"
    params = {
        "fields": "id,message,from,created_time,like_count,comment_count,parent,attachment",
        "limit": limit
    }
    if since: params["since"] = since
    if until: params["until"] = until

    comments = []
    while True:
        res = _get(url, params)
        comments.extend(res.get("data", []))
        next_url = res.get("paging", {}).get("next")
        if not next_url:
            break
        url = next_url
        params = {}

    print(f"✅ {len(comments)} comments fetched for post {post_id}")
    return comments


# ======================================================
# INSTAGRAM INSIGHTS
# ======================================================
def get_instagram_metrics(metric_set="reach,impressions,saved,engagement", period="day"):
    """Instagram metrics such as reach, impressions, engagement."""
    url = f"{GRAPH}/{IG_ID}/insights"
    params = {"metric": metric_set, "period": period}
    return _get(url, params)


# ======================================================
# ADS INSIGHTS
# ======================================================
def get_ads_insights(level="campaign", fields=None,
                       date_preset="last_30d", breakdowns=None, time_increment=1):
    """Facebook Ads Insights (campaign spend, reach, clicks, conversions)."""
    url = f"{GRAPH}/act_{AD_ACCOUNT_ID}/insights"
    params = {
        "level": level,
        "date_preset": date_preset,
        "time_increment": time_increment,
        "fields": fields or "campaign_name,spend,impressions,reach,clicks,actions,conversions",
        "limit": 200,
    }
    if breakdowns:
        params["breakdowns"] = breakdowns
    return _get(url, params)


# ======================================================
# DATA EXPORT (FIXED TO /data/downloads)
# ======================================================
def export_to_csv(data: List[Dict[str, Any]], filename="fb_data.csv"):
    """Export data to CSV in /data/downloads."""
    if not data:
        print("⚠️ No data to export.")
        return
    out_path = DOWNLOADS_DIR / filename
    keys = sorted({k for d in data for k in d.keys()})
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
    print(f"💾 Exported {len(data)} rows → {out_path}")
    return str(out_path)


def save_to_json(data: Any, filename="fb_data.json"):
    """Save data to JSON in /data/downloads."""
    out_path = DOWNLOADS_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved {out_path}")
    return str(out_path)


from pathlib import Path

# Төслийн үндсэн хавтас руу заах
BASE_DIR = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = BASE_DIR / "data" / "downloads"

def export_to_xlsx(data: List[Dict[str, Any]], filename: str):
    """Export ANY dataset to XLSX inside /data/downloads/"""
    df = pd.DataFrame(data)
    if df.empty:
        print("⚠️ No data for Excel export.")
        return None
    
    # Ensure folder exists
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOWNLOADS_DIR / filename
    df.to_excel(out_path, index=False)
    print(f"📘 Saved {len(df)} rows → {out_path}")
    
    # Flask route-д зөв ажиллах зам буцаах
    return f"/data/downloads/{filename}"



# ======================================================
# WRAPPER: DATE-RANGE HARVESTER
# ======================================================
def harvest_marketing_data(mode="page_posts", **kwargs):
    """
    mode options:
      - page_insights
      - page_posts
      - post_comments
      - ig_insights
      - ads_insights
    """
    if mode == "page_insights":
        return get_page_insights()

    elif mode == "page_posts":
        return list_posts(kwargs.get("since"), kwargs.get("until"), limit=kwargs.get("limit", 20))

    elif mode == "post_comments":
        pid = kwargs.get("post_id")
        since = kwargs.get("since")
        until = kwargs.get("until")

        if not pid:
            posts = list_posts(limit=3)
            if not posts:
                print("⚠️ No posts found to fetch comments.")
                return []
            pid = posts[0]["id"]

        return list_comments(pid, since=since, until=until)

    elif mode == "ig_insights":
        return get_instagram_metrics()

    elif mode == "ads_insights":
        return get_ads_insights(**kwargs)

    else:
        raise ValueError("Invalid mode specified.")


# ======================================================
# TEST
# ======================================================
if __name__ == "__main__":
    print("🔍 Testing fb_fetcher.py (Unitel AI Hub)")

    if not ACCESS_TOKEN:
        print("⚠️ FB_ACCESS_TOKEN missing in .env")
        exit()

    SINCE = "2025-10-01"
    UNTIL = "2025-11-01"

    print("\n=== POSTS ===")
    posts = list_posts(since=SINCE, until=UNTIL, limit=10)
    export_to_xlsx(posts, "fb_posts_test.xlsx")

    if posts:
        pid = posts[0]["id"]
        print(f"\n=== COMMENTS for post {pid} ===")
        comments = list_comments(pid, since=SINCE, until=UNTIL)
        export_to_xlsx(comments, "fb_comments_test.xlsx")

    print("\n=== ADS INSIGHTS ===")
    ads = get_ads_insights(date_preset="last_30d")
    export_to_xlsx(ads.get("data", []), "fb_ads_test.xlsx")

    print("\n✅ Done. Check /data/downloads/")