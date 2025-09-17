from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests
from dotenv import load_dotenv

from common import BannerDeduper, BannerRecord
from gogo_mn import scrape_gogo
from ikon_mn import scrape_ikon
from news_mn import scrape_news

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("scraper")

HEADLESS = os.getenv("HEADLESS", "1") != "0"
DEFAULT_DWELL = int(os.getenv("DWELL_DEFAULT", "35"))
MIN_SCORE = int(os.getenv("AD_MIN_SCORE", "3"))

SCRAPER_OUTPUT = Path(os.getenv("SCRAPER_SHOT_DIR", "banner_screenshots"))
DATE_DIR = SCRAPER_OUTPUT / datetime.utcnow().strftime("%Y-%m-%d")
DATE_DIR.mkdir(parents=True, exist_ok=True)

INGEST_BASE = os.getenv("INGEST_BASE", "http://127.0.0.1:8888").rstrip("/")
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "")
UPLOAD_URL = f"{INGEST_BASE}/ads/api/upload"
INGEST_URL = f"{INGEST_BASE}/ads/api/ingest"

session = requests.Session()
headers = {"User-Agent": "UnitelScraper/1.0"}
if INGEST_TOKEN:
    headers["X-INGEST-TOKEN"] = INGEST_TOKEN
json_headers = {**headers, "Content-Type": "application/json"}

SiteScraper = Callable[[Path, int, bool], List[Dict[str, Any]]]
SITES: List[Dict[str, Any]] = [
    {
        "name": "gogo.mn",
        "fn": scrape_gogo,
        "dwell": int(os.getenv("DWELL_GOGO", str(DEFAULT_DWELL))),
        "min_score": int(os.getenv("MIN_SCORE_GOGO", str(MIN_SCORE))),
    },
    {
        "name": "ikon.mn",
        "fn": scrape_ikon,
        "dwell": int(os.getenv("DWELL_IKON", "55")),
        "min_score": int(os.getenv("MIN_SCORE_IKON", str(max(1, MIN_SCORE - 1)))),
    },
    {
        "name": "news.mn",
        "fn": scrape_news,
        "dwell": int(os.getenv("DWELL_NEWS", str(DEFAULT_DWELL))),
        "min_score": int(os.getenv("MIN_SCORE_NEWS", str(MIN_SCORE))),
    },
]


def upload_screenshot(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    with file_path.open("rb") as fh:
        files = {"file": (file_path.name, fh, "image/png")}
        try:
            resp = session.post(UPLOAD_URL, files=files, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data.get("url")
        except Exception as exc:
            logger.error("upload failed for %s: %s", file_path, exc)
            return None


def send_ingest(record: BannerRecord, screenshot_url: Optional[str]) -> bool:
    payload = record.as_ingest_payload(screenshot_url)
    try:
        resp = session.post(INGEST_URL, json=payload, headers=json_headers, timeout=60)
        resp.raise_for_status()
        return True
    except Exception as exc:
        logger.error("ingest failed for %s (%s): %s", record.site, record.ad_id, exc)
        return False


def main() -> int:
    deduper = BannerDeduper(distance_threshold=5)
    total_sent = 0
    for site in SITES:
        name = str(site["name"])
        scraper: SiteScraper = site["fn"]  # type: ignore[assignment]
        dwell = int(site["dwell"])
        min_score = int(site["min_score"])

        logger.info("Scraping %s (dwell=%ss, min_score=%s)", name, dwell, min_score)
        try:
            captures = scraper(DATE_DIR, dwell_seconds=dwell, headless=HEADLESS)
        except Exception as exc:
            logger.exception("scrape failed for %s: %s", name, exc)
            continue

        logger.info("%s returned %d candidates", name, len(captures))
        sent_site = 0
        for capture in captures:
            record = BannerRecord.from_capture(capture, min_score=min_score)
            if record is None:
                continue
            if record.is_ad != "1":
                logger.debug("skip %s score=%s (%s)", record.src, record.score, record.reason)
                continue
            if not deduper.add(record):
                logger.debug("dedupe filtered %s", record.src)
                continue
            screenshot_url = upload_screenshot(record.screenshot_path)
            if send_ingest(record, screenshot_url):
                sent_site += 1
                total_sent += 1
                logger.info(
                    "INGEST %s | %s | score=%s | %s",
                    record.site,
                    record.ad_id,
                    record.score,
                    record.reason,
                )
        logger.info("%s: sent %d new ads", name, sent_site)
    logger.info("Scraper completed. total_ads=%d", total_sent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
