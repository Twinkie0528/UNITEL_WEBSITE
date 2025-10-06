from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

from common import BannerDeduper, BannerRecord, canonical_src

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("scraper")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


DEFAULT_DWELL = _env_int("DWELL_DEFAULT", 30)
MIN_SCORE_DEFAULT = _env_int("AD_MIN_SCORE", 3)
ACTIVE_WINDOW_DAYS = _env_int("ACTIVE_WINDOW_DAYS", 1)
SUMMARY_DAYS = _env_int("SUMMARY_DAYS", 0)
PHASH_DISTANCE = _env_int("PHASH_DISTANCE", 5)
RESET_EVENTS = os.getenv("RESET_EVENTS", "0") == "1"

ROOT_DIR = Path(__file__).resolve().parent
SCRAPER_OUTPUT = ROOT_DIR / "banner_screenshots"
EXPORT_DIR = ROOT_DIR / "_export"
SCRAPER_OUTPUT.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
DATE_DIR = SCRAPER_OUTPUT / datetime.now(timezone.utc).strftime("%Y-%m-%d")
DATE_DIR.mkdir(parents=True, exist_ok=True)

EVENTS_PATH = EXPORT_DIR / "events.jsonl"
TSV_OUT = EXPORT_DIR / "ads.tsv"
XLSX_OUT = EXPORT_DIR / "ads.xlsx"

DISABLE_INGEST = os.getenv("DISABLE_INGEST", "1") == "1"
INGEST_BASE = os.getenv("INGEST_BASE", "http://127.0.0.1:8888").rstrip("/")
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "")
UPLOAD_URL = f"{INGEST_BASE}/ads/api/upload"
INGEST_URL = f"{INGEST_BASE}/ads/api/ingest"

session = requests.Session()
headers = {"User-Agent": "UnitelScraper/2.0"}
if INGEST_TOKEN:
    headers["X-INGEST-TOKEN"] = INGEST_TOKEN
json_headers = {**headers, "Content-Type": "application/json"}

SiteScraper = Callable[[Path], List[Dict[str, Any]]]

SITE_REGISTRY: List[Dict[str, Any]] = [
    {
        "name": "gogo.mn",
        "slug": "gogo.mn",
        "module": "sites.gogo_mn",
        "callable": "scrape_gogo",
        "dwell_env": "DWELL_GOGO",
        "score_env": "MIN_SCORE_GOGO",
    },
    {
        "name": "ikon.mn",
        "slug": "ikon.mn",
        "module": "sites.ikon_mn",
        "callable": "scrape_ikon",
        "dwell_env": "DWELL_IKON",
        "score_env": "MIN_SCORE_IKON",
    },
    {
        "name": "news.mn",
        "slug": "news.mn",
        "module": "sites.news_mn",
        "callable": "scrape_news",
        "dwell_env": "DWELL_NEWS",
        "score_env": "MIN_SCORE_NEWS",
    },
    {
        "name": "bolor-toli.com",
        "slug": "bolor-toli.com",
        "module": "sites.bolortoli_com",
        "callable": "scrape_bolortoli",
        "dwell_env": "DWELL_BOLORTOLI",
        "score_env": "MIN_SCORE_BOLORTOLI",
    },
    {
        "name": "caak.mn",
        "slug": "caak.mn",
        "module": "sites.caak_mn",
        "callable": "scrape_caak",
        "dwell_env": "DWELL_CAAK",
        "score_env": "MIN_SCORE_CAAK",
    },
    {
        "name": "ublife.mn", 
         "module": "sites.ublife_pw", 
         "callable": "scrape_ublife",
        "dwell_env": "DWELL_UBLIFE", 
        "score_env": "MIN_SCORE_UBLIFE", 
        "dwell_default": 60,
    },
    {
        "name": "lemonpress.mn",
        "slug": "lemonpress.mn",
        "module": "sites.lemonpress_mn",
        "callable": "scrape_lemonpress",
        "dwell_env": "DWELL_LEMONPRESS",
        "score_env": "MIN_SCORE_LEMONPRESS",
    },
]


def _load_scraper(module_name: str, func_name: str) -> Optional[SiteScraper]:
    try:
        module = import_module(module_name)
    except ModuleNotFoundError:
        logger.warning("Module missing: %s (site will be skipped)", module_name)
        return None
    fn = getattr(module, func_name, None)
    if not callable(fn):
        logger.warning("Callable missing: %s.%s (site will be skipped)", module_name, func_name)
        return None
    return fn  # type: ignore[return-value]


def upload_screenshot(path: Optional[str]) -> Optional[str]:
    if DISABLE_INGEST or not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        with file_path.open("rb") as fh:
            files = {"file": (file_path.name, fh, "image/png")}
            response = session.post(UPLOAD_URL, files=files, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json().get("url")
    except Exception as exc:
        logger.error("Upload failed for %s: %s", file_path, exc)
        return None


def send_ingest(record: BannerRecord, screenshot_url: Optional[str]) -> bool:
    if DISABLE_INGEST:
        return True
    try:
        payload = record.as_ingest_payload(screenshot_url)
        response = session.post(INGEST_URL, json=payload, headers=json_headers, timeout=60)
        response.raise_for_status()
        return True
    except Exception as exc:
        logger.error("Ingest failed for %s (%s): %s", record.site, record.ad_id, exc)
        return False


def _event_write(record: BannerRecord, screenshot_url: Optional[str]) -> None:
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ad_id": record.ad_id,
        "site": record.site,
        "score": record.score,
        "reason": record.reason,
        "src": record.src,
        "landing_url": record.landing_final_url or record.landing_url,
        "landing_original_url": record.landing_url,
        "landing_final_url": record.landing_final_url,
        "width": record.width,
        "height": record.height,
        "context": record.context,
        "phash": record.phash_hex or "",
        "screenshot_path": record.screenshot_path,
        "screenshot_url": screenshot_url or "",
    }
    with EVENTS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_events_jsonl() -> List[Dict[str, Any]]:
    if not EVENTS_PATH.exists():
        return []
    events: List[Dict[str, Any]] = []
    with EVENTS_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events


AD_NETWORKS = (
    "boost.mn",
    "exchange.boost.mn",
    "googlesyndication.com",
    "doubleclick.net",
    "taboola.com",
    "mgid.com",
    "criteo.com",
    "adnxs.com",
)

PUBLISHER_HOSTS = (
    "gogo.mn",
    "ikon.mn",
    "news.mn",
    "caak.mn",
    "bolor-toli.com",
    "lemonpress.mn",
    "ublife.mn",
    "ub.life",
)

BRAND_MAP = {
    "khanbank.com": "Khan Bank",
    "tdbm.mn": "TDB",
    "golomtbank.com": "Golomt Bank",
    "xacbank.mn": "XacBank",
    "bogdbank.com": "Bogd Bank",
    "statebank.mn": "State Bank",
    "unitel.mn": "Unitel",
    "mobicom.mn": "Mobicom",
    "skytel.mn": "Skytel",
    "g-mobile.mn": "G-Mobile",
    "pcmall.mn": "PC Mall",
    "nomin.mn": "Nomin",
    "emartmongolia.mn": "E-Mart",
    "koreanair.com": "Korean Air",
}


def _host_core(url: str) -> str:
    try:
        parsed = urlparse(url)
    except Exception:
        return ""
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    parts = [part for part in host.split(".") if part]
    if len(parts) >= 3:
        return ".".join(parts[-3:])
    return host


def _brand_from_url(url: str) -> str:
    if not url:
        return ""
    core = _host_core(url)
    if not core or core in PUBLISHER_HOSTS:
        return ""
    if any(core.endswith(network) for network in AD_NETWORKS):
        return ""
    if core in BRAND_MAP:
        return BRAND_MAP[core]
    base = core.split(".")[0]
    cleaned = base.replace("-", " ").replace("_", " ").strip()
    lower = cleaned.lower()
    for suffix in ("bank", "insurance", "air", "airlines", "mall", "shop", "store"):
        if lower.endswith(suffix) and f" {suffix}" not in lower:
            idx = lower.rfind(suffix)
            cleaned = cleaned[:idx] + " " + cleaned[idx:]
            break
    brand = " ".join(word.capitalize() for word in cleaned.split())
    if brand.lower() == "tdbm":
        return "TDB"
    return brand


def _file_uri(path: str) -> str:
    if not path:
        return ""
    try:
        return Path(path).resolve().as_uri()
    except Exception:
        return ""


def _excel_hyperlink(url: str) -> str:
    if not url:
        return ""
    safe = url.replace('"', '""')
    return f'=HYPERLINK("{safe}","{safe}")'


def _write_summary(events: List[Dict[str, Any]]) -> Tuple[int, int]:
    def _to_date(value: str) -> Optional[datetime.date]:
        try:
            dt = datetime.fromisoformat(value.replace("Z", ""))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.date()
        except Exception:
            return None

    if SUMMARY_DAYS > 0:
        today = datetime.now(timezone.utc).date()
        cutoff = today - timedelta(days=SUMMARY_DAYS - 1)
        events = [event for event in events if (d := _to_date(event.get("seen_at", ""))) and d >= cutoff]

    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for event in events:
        site = event.get("site") or ""
        src = event.get("src") or ""
        key_src = canonical_src(src) or (event.get("ad_id") or "")
        if not site or not key_src:
            continue
        key = (site, key_src)
        seen_at = event.get("seen_at", "")
        seen_date = _to_date(seen_at)
        landing_final = event.get("landing_final_url") or event.get("landing_url") or ""
        landing_raw = event.get("landing_original_url") or ""

        if key not in groups:
            groups[key] = {
                "site": site,
                "first": seen_at,
                "last": seen_at,
                "dates": set([seen_date] if seen_date else []),
                "times": 1,
                "score": event.get("score", 0),
                "reason": event.get("reason", ""),
                "src": src,
                "landing": landing_final or "",
                "landing_raw": landing_raw,
                "shot_path": event.get("screenshot_path", ""),
                "shot_url": event.get("screenshot_url", ""),
                "context": event.get("context", ""),
            }
        else:
            bucket = groups[key]
            bucket["times"] += 1
            if seen_date:
                bucket["dates"].add(seen_date)
            if seen_at and seen_at < bucket["first"]:
                bucket["first"] = seen_at
            if seen_at and seen_at > bucket["last"]:
                bucket["last"] = seen_at
            if event.get("score", 0) > bucket["score"]:
                bucket["score"] = event.get("score", 0)
            if event.get("reason"):
                bucket["reason"] = event.get("reason")
            if src:
                bucket["src"] = src
            if landing_final or landing_raw:
                bucket["landing"] = landing_final or landing_raw
                bucket["landing_raw"] = landing_raw or bucket["landing_raw"]
            if event.get("screenshot_path"):
                bucket["shot_path"] = event.get("screenshot_path")
            if event.get("screenshot_url"):
                bucket["shot_url"] = event.get("screenshot_url")
            if event.get("context"):
                bucket["context"] = event.get("context")

    today = datetime.now(timezone.utc).date()
    active_cutoff = today - timedelta(days=max(1, ACTIVE_WINDOW_DAYS) - 1)

    rows: List[Dict[str, Any]] = []
    for bucket in groups.values():
        first_date = _to_date(bucket["first"]) if bucket.get("first") else None
        last_date = _to_date(bucket["last"]) if bucket.get("last") else None
        status = "ИДЭВХТЭЙ" if (last_date and last_date >= active_cutoff) else "ДУУССАН"

        landing_candidate = bucket.get("landing") or bucket.get("landing_raw") or bucket.get("src") or ""
        brand = _brand_from_url(landing_candidate)
        if not brand and bucket.get("context"):
            context = str(bucket.get("context"))
            for part in context.split():
                if part.startswith("adv="):
                    brand = _brand_from_url(part.split("=", 1)[1])
                    if brand:
                        break

        rows.append(
            {
                "site": bucket["site"],
                "status": status,
                "first_seen_date": first_date.isoformat() if first_date else "",
                "last_seen_date": last_date.isoformat() if last_date else "",
                "days_seen": len([d for d in bucket["dates"] if d]),
                "times_seen": bucket["times"],
                "ad_score": bucket["score"],
                "ad_reason": str(bucket["reason"]).replace(" | ", ","),
                "brand": brand,
                "src": bucket["src"],
                "src_open": bucket["src"],
                "landing_url": bucket.get("landing", ""),
                "landing_open": bucket.get("landing", ""),
                "screenshot_path": bucket.get("shot_path", ""),
                "screenshot_file": bucket.get("shot_url") or _file_uri(bucket.get("shot_path", "")),
            }
        )

    fieldnames = [
        "site",
        "status",
        "first_seen_date",
        "last_seen_date",
        "days_seen",
        "times_seen",
        "ad_score",
        "ad_reason",
        "brand",
        "src",
        "src_open",
        "landing_url",
        "landing_open",
        "screenshot_path",
        "screenshot_file",
    ]

    with TSV_OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    xlsx_written = 0
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows, columns=fieldnames)
        for column in ("src", "src_open", "landing_url", "landing_open", "screenshot_file"):
            if column in df.columns:
                df[column] = df[column].apply(_excel_hyperlink)

        temp_path = XLSX_OUT.with_suffix(".tmp.xlsx")
        try:
            df.to_excel(temp_path, index=False)
            os.replace(temp_path, XLSX_OUT)
            xlsx_written = 1
        except PermissionError:
            alt = EXPORT_DIR / f"ads_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.xlsx"
            df.to_excel(alt, index=False)
            xlsx_written = 1
            logger.warning("XLSX locked (open in Excel?). Wrote to %s", alt)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("XLSX export skipped: %s", exc)

    logger.info("SUMMARY: wrote _export/ads.tsv (%d rows) and _export/ads.xlsx", len(rows))
    return len(rows), xlsx_written


def main() -> int:
    if RESET_EVENTS and EVENTS_PATH.exists():
        try:
            EVENTS_PATH.unlink()
            logger.info("events.jsonl reset (RESET_EVENTS=1)")
        except Exception as exc:
            logger.warning("Could not reset events.jsonl: %s", exc)

    deduper = BannerDeduper(distance_threshold=PHASH_DISTANCE)
    total_accepted = 0

    for site in SITE_REGISTRY:
        name = site["name"]
        slug = site["slug"]
        dwell = _env_int(site.get("dwell_env", ""), DEFAULT_DWELL)
        min_score = _env_int(site.get("score_env", ""), MIN_SCORE_DEFAULT)

        scraper_func = _load_scraper(site["module"], site["callable"])
        if scraper_func is None:
            continue

        logger.info("Scraping %s (dwell=%ss, min_score=%s)", name, dwell, min_score)
        site_output_dir = DATE_DIR / slug
        try:
            captures = scraper_func(site_output_dir) or []
        except Exception as exc:  # pragma: no cover - network dependent
            logger.exception("scrape failed for %s: %s", name, exc)
            continue

        logger.info("%s returned %d candidates", name, len(captures))
        if not captures:
            logger.warning("%s produced 0 candidates - check selectors or fallback.", name)

        accepted_site = 0
        for capture in captures:
            record = BannerRecord.from_capture(capture, min_score=min_score)
            if record is None or record.is_ad != "1":
                continue
            if not deduper.add(record):
                continue

            screenshot_url = upload_screenshot(record.screenshot_path)
            _event_write(record, screenshot_url)

            if send_ingest(record, screenshot_url):
                accepted_site += 1
                total_accepted += 1
                logger.info(
                    "ACCEPT %s | %s | score=%s | reason=%s",
                    record.site,
                    record.ad_id,
                    record.score,
                    record.reason,
                )

        logger.info("%s: accepted %d ads", name, accepted_site)

    events = _read_events_jsonl()
    _write_summary(events)
    logger.info("Scraper completed. total_accepted=%d", total_accepted)
    if DISABLE_INGEST:
        logger.info("Local mode: ingest/upload are disabled (DISABLE_INGEST=1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
