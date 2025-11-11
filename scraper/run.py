# -*- coding: utf-8 -*-
# scraper/run.py — САЙЖРУУЛСАН ХУВИЛБАР (Зэрэгцээ ажиллагаатай)
import os
import json
import time
import traceback
import logging
import threading
from datetime import datetime
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ---- .env сонголттой ----
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded config from .env file.")
except Exception:
    print("[WARN] python-dotenv not found. Skipping .env file.")

from common import ensure_dir, load_db, save_db, upsert_banner, BannerRecord

# --- Сайтын скраперууд ---
from gogo_mn import scrape_gogo
from ikon_mn import scrape_ikon
from news_mn import scrape_news
from caak_mn import scrape_caak
from bolortoli_mn import scrape_bolortoli
from ublife_mn import scrape_ublife
from lemonpress_mn import scrape_lemonpress

# =========================
# Config (ENV with defaults)
# =========================
def _b(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")

def _i(name: str, default: int) -> int:
    try: return int(os.getenv(name, default))
    except Exception: return default

def _s(name: str, default: str) -> str:
    return os.getenv(name, default)

# Scraping тохиргоо
HEADLESS        = _b("HEADLESS", True)
ADS_ONLY        = _b("ADS_ONLY", True)
MAX_WORKERS     = _i("MAX_WORKERS", 4) # Зэрэг ажиллах scraper-ийн тоо

# Глобал босго (сайт тус бүрийн ENV байхгүй бол эдгээр унаган утга хэрэглэнэ)
DWELL_SEC       = _i("DWELL_SEC", 35)
ADS_MIN_SCORE   = _i("ADS_MIN_SCORE", 3)

# <<< ЗАСВАРЛАСАН ХЭСЭГ >>>
# Энэ хэсэгт түүхий өгөгдөл хадгалах `banner_tracking_combined.tsv` замыг зааж өгнө.
# Ингэснээр summarize.py скрипт энэ файлаас уншиж ажиллах боломжтой болно.
OUT_DIR         = os.path.abspath(_s("OUT_DIR", "./banner_screenshots"))
TSV_PATH        = os.path.abspath(_s("TSV_PATH", "./banner_tracking_combined.tsv"))

# Backend ingest endpoints (Одоогоор идэвхгүй)
#INGEST_BASE     = _s("INGEST_BASE", "http://127.0.0.1:8888/").rstrip("/")
# ... (бусад ingest-ийн тохиргоо) ...

# Logging
LOG_LEVEL = _s("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ... (HTTP болон Ingest-ийн туслах функцууд энд хэвээрээ байна) ...
# (Эдгээр функц дуудагдахгүй тул асуудал үүсгэхгүй)

# =========================
# Site config helper (ENV override)
# =========================
def site_cfg(key: str) -> tuple[int, int]:
    """Сайт тус бүрийн тохиргоог .env-ээс унших."""
    dwell = _i(f"{key}_DWELL", DWELL_SEC)
    msc   = _i(f"{key}_MIN_SCORE", ADS_MIN_SCORE)
    return dwell, msc

# =========================
# Main
# =========================
def main():
    date_dir = os.path.join(OUT_DIR, datetime.now().strftime("%Y-%m-%d"))
    ensure_dir(date_dir)
    logging.info("🚀 Scan start → %s", date_dir)

    # --- САЙЖРУУЛАЛТ: Сайтын тохиргоог нэгдсэн, цэгцтэй жагсаалт болгосон ---
    SITES_CONFIG = [
        {"key": "GOGO",       "name": "gogo",       "fn": scrape_gogo},
        {"key": "IKON",       "name": "ikon",       "fn": scrape_ikon},
        {"key": "NEWS",       "name": "news",       "fn": scrape_news},
        {"key": "CAAK",       "name": "caak",       "fn": scrape_caak},
        {"key": "BOLORTOLI",  "name": "bolor-toli", "fn": scrape_bolortoli},
        {"key": "UBLIFE",     "name": "ublife",     "fn": scrape_ublife},
        {"key": "LEMONPRESS", "name": "lemonpress", "fn": scrape_lemonpress},
    ]

    rows = load_db(TSV_PATH)
    # --- САЙЖРУУЛАЛТ: Олон thread зэрэг хандах үед өгөгдлийг хамгаалах Lock ---
    rows_lock = threading.Lock()
    
    total_new = 0
    total_seen = 0

    logging.info(f"Starting thread pool with {MAX_WORKERS} workers.")

    # --- САЙЖРУУЛАЛТ: ThreadPoolExecutor ашиглан scraper-уудыг зэрэг ажиллуулах ---
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_site = {}
        for config in SITES_CONFIG:
            dwell, min_score = site_cfg(config["key"])
            logging.info(f"▶ Queuing {config['name']} (dwell={dwell}, min_score={min_score})")
            future = executor.submit(
                config["fn"],
                output_dir=date_dir,
                dwell_seconds=dwell,
                headless=HEADLESS,
                ads_only=ADS_ONLY,
                min_score=min_score,
            )
            future_to_site[future] = config["name"]

        for future in as_completed(future_to_site):
            site_name = future_to_site[future]
            try:
                caps = future.result()
                logging.info(f"🔎 COMPLETED {site_name}: found {len(caps)} candidates.")
                
                site_new_count = 0
                for cap in caps:
                    rec = BannerRecord.from_capture(cap, min_ad_score=ADS_MIN_SCORE)
                    
                    # --- Lock ашиглан нэгдсэн `rows` жагсаалтыг хамгаалах ---
                    with rows_lock:
                        changed, inserted = upsert_banner(rows, rec)
                        total_seen += 1
                        if inserted:
                            total_new += 1
                            site_new_count += 1
                    
                    if inserted:
                        # --- САЙЖРУУЛАЛТ: Логт landing_url-г нэмсэн ---
                        logging.info(
                            "[NEW] %-12s | ad:%s score:%s | landing: %s | src: %s",
                            rec.site, rec.is_ad, rec.ad_score, rec.landing_url, rec.src
                        )

                    if not inserted and rec.screenshot_path and os.path.exists(rec.screenshot_path):
                        try:
                            os.remove(rec.screenshot_path)
                        except Exception: pass
                
                if site_new_count > 0:
                    logging.info(f"📈 {site_name}: Added {site_new_count} new banners to the database.")

            except Exception as e:
                logging.error(f"FATAL error during {site_name} execution: {e}")
                traceback.print_exc()

    save_db(TSV_PATH, rows)
    logging.info("💾 TSV saved: %s (rows=%d)", TSV_PATH, len(rows))
    logging.info("📊 Summary: total_seen=%d | total_new=%d", total_seen, total_new)
    logging.info("✅ Done.")

if __name__ == "__main__":
    main()