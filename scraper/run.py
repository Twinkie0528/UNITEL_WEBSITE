# -*- coding: utf-8 -*-
# scraper/run.py — scrape gogo/ikon/news → TSV + backend ingest
import os, traceback, logging, requests
from datetime import datetime
from dotenv import load_dotenv

from common import ensure_dir, load_db, save_db, upsert_banner, BannerRecord
from gogo_mn import scrape_gogo
from ikon_mn import scrape_ikon
from news_mn import scrape_news

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------- Config ----------
HEADLESS     = True
DWELL_SEC    = 35
ADS_ONLY     = True        # (өмнө нь зарлагдаагүй байсан)
ADS_MIN_SCORE = 2

IKON_DWELL   = 55
IKON_MIN_SCORE = 1

OUT_DIR  = os.path.abspath("./banner_screenshots")
TSV_PATH = os.path.abspath("./banner_tracking_combined.tsv")

# Backend ingest endpoints
INGEST_BASE  = os.getenv("INGEST_BASE", "http://127.0.0.1:8888").rstrip("/")
API_URL      = os.getenv("API_URL", f"{INGEST_BASE}/ads/api/ingest")
UPLOAD_URL   = f"{INGEST_BASE}/ads/api/upload"
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "")

HEADERS_JSON = {"Content-Type": "application/json", **({"X-INGEST-TOKEN": INGEST_TOKEN} if INGEST_TOKEN else {})}
HEADERS_FORM = ({"X-INGEST-TOKEN": INGEST_TOKEN} if INGEST_TOKEN else {})

# ---------- Helpers ----------
def _upload_screenshot(local_path: str) -> str | None:
    """/ads/api/upload руу screenshot илгээнэ → /static/... URL буцаана."""
    try:
        if not local_path or not os.path.exists(local_path):
            return None
        with open(local_path, "rb") as f:
            files = {"file": (os.path.basename(local_path), f, "image/png")}
            r = requests.post(UPLOAD_URL, files=files, headers=HEADERS_FORM, timeout=30)
            r.raise_for_status()
            return (r.json() or {}).get("url")
    except Exception as e:
        logging.error("upload fail: %s", e)
        return None

def send_to_ingest_api(record: BannerRecord, shot_url: str | None):
    """
    Flask backend /ads/api/ingest руу илгээнэ.
    app.py тал 'url' (landing) талбарыг ашигладаг тул landing_open биш 'url' илгээнэ.
    """
    if record.is_ad != "1":
        return
    payload = {
        "ad_id": record.ad_id,
        "site": record.site,
        "status": "active",
        "url": record.landing_url,    # !!! brand-д хэрэгтэй
        "src_open": record.src,
        "screenshot": (shot_url or "")
    }
    try:
        resp = requests.post(API_URL, headers=HEADERS_JSON, json=payload, timeout=30)
        resp.raise_for_status()
        logging.info("ingest ok: %s | %s", record.site, record.ad_id)
    except requests.exceptions.RequestException as e:
        logging.error("ingest fail: %s | %s", record.site, e)

# ---------- Main ----------
date_dir = os.path.join(OUT_DIR, datetime.now().strftime("%Y-%m-%d"))
ensure_dir(date_dir)
logging.info("🚀 Scan start → %s", date_dir)

sites = [
    ("gogo", scrape_gogo, DWELL_SEC, ADS_MIN_SCORE),
    ("ikon", scrape_ikon, IKON_DWELL, IKON_MIN_SCORE),
    ("news", scrape_news, DWELL_SEC, ADS_MIN_SCORE),
]

rows = load_db(TSV_PATH)

for name, fn, dwell, min_score in sites:
    try:
        caps = fn(
            date_dir,
            dwell_seconds=dwell,
            headless=HEADLESS,
            ads_only=ADS_ONLY,
            min_score=min_score,
        )
        logging.info("🔎 %s: %d candidates", name, len(caps))

        for cap in caps:
            rec = BannerRecord.from_capture(cap, min_ad_score=min_score)
            changed, inserted = upsert_banner(rows, rec)
            if inserted:
                logging.info("[NEW] %s → %s | is_ad:%s score:%s",
                             rec.site, rec.src or rec.screenshot_path, rec.is_ad, rec.ad_score)

            # upload screenshot (optional)
            shot_url = _upload_screenshot(rec.screenshot_path) if rec.screenshot_path else None
            # push to backend
            send_to_ingest_api(rec, shot_url)

            # not inserted бол локал скриншотыг устгаж болно (хэмнэлт)
            if not inserted and rec.screenshot_path and os.path.exists(rec.screenshot_path):
                try: os.remove(rec.screenshot_path)
                except Exception: pass

    except Exception as e:
        logging.error("FATAL %s: %s", name, e)
        traceback.print_exc()

save_db(TSV_PATH, rows)
logging.info("💾 TSV saved: %s (rows=%d)", TSV_PATH, len(rows))
logging.info("✅ Done.")
