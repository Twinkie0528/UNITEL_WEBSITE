from __future__ import annotations

import csv
import io
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, render_template, request, send_file
from flask_cors import CORS
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

import processor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

ROOT_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = ROOT_DIR / "templates"
STATIC_DIR = ROOT_DIR / "static"
SCRAPER_DIR = ROOT_DIR / "scraper"
BANNER_SCREENSHOT_DIR = SCRAPER_DIR / "banner_screenshots"
UPLOAD_ROOT = STATIC_DIR / "ads"

for directory in (STATIC_DIR, UPLOAD_ROOT, BANNER_SCREENSHOT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
app.config["JSON_SORT_KEYS"] = False
CORS(app)

PORT = int(os.getenv("PORT", "8888"))
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "unitel_aihub")
ADS_COLLECTION_NAME = os.getenv("ADS_COLLECTION", "ads_events")
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "")
RESET_TOKEN = os.getenv("RESET_TOKEN", "")
SCRAPER_TIMEOUT = int(os.getenv("SCRAPER_TIMEOUT", "900"))

mongo_client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
try:
    mongo_client.admin.command("ping")
    logging.info("Connected to MongoDB at %s", MONGO_URL)
except Exception as exc:
    logging.exception("MongoDB connection failed: %s", exc)
    raise

db = mongo_client[DB_NAME]
ads_collection: Collection = db[ADS_COLLECTION_NAME]
ads_collection.create_index(
    [
        ("ad_id", ASCENDING),
        ("site", ASCENDING),
        ("url", ASCENDING),
        ("detected_at", DESCENDING),
    ],
    name="ads_lookup",
    background=True,
)
ads_collection.create_index([("detected_at", DESCENDING)], name="detected_at_desc", background=True)

REPORT_HEADERS = [
    "ad_id",
    "site",
    "brand",
    "first_seen_date",
    "last_seen_date",
    "days_seen",
    "times_seen",
    "status",
    "src_open",
    "landing_open",
    "screenshot",
]

BRAND_HOST_MAP: Dict[str, str] = {
    "unitel.mn": "Unitel",
    "www.unitel.mn": "Unitel",
    "shop.unitel.mn": "Unitel",
    "skytel.mn": "Skytel",
    "www.skytel.mn": "Skytel",
    "mobicom.mn": "Mobicom",
    "www.mobicom.mn": "Mobicom",
    "gmobile.mn": "G-Mobile",
    "www.gmobile.mn": "G-Mobile",
    "ardcredit.mn": "ARD Credit",
    "www.ardcredit.mn": "ARD Credit",
    "ard.mn": "ARD Bank",
    "www.ard.mn": "ARD Bank",
    "khanbank.com": "Khan Bank",
    "www.khanbank.com": "Khan Bank",
    "golomtbank.com": "Golomt Bank",
    "www.golomtbank.com": "Golomt Bank",
    "bogdbank.com": "Bogd Bank",
    "www.bogdbank.com": "Bogd Bank",
}

BRAND_KEYWORDS: Dict[str, str] = {
    "unitel": "Unitel",
    "skytel": "Skytel",
    "mobicom": "Mobicom",
    "gmobile": "G-Mobile",
    "ardcredit": "ARD Credit",
    "ardbank": "ARD Bank",
    "ard": "ARD",
    "khanbank": "Khan Bank",
    "golomt": "Golomt Bank",
    "bogd": "Bogd Bank",
    "tavanbogd": "Таван Богд",
    "xacbank": "Хас Банк",
}

scrape_lock = Lock()
scrape_thread: Optional[Thread] = None
scrape_state: Dict[str, Any] = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "error": None,
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_date(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).date().isoformat()


def host_from_url(url: str) -> str:
    if not url:
        return ""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def infer_brand(url: str, fallback: Optional[str] = None) -> str:
    host = host_from_url(url)
    if not host:
        return fallback or ""
    host = host.split(":")[0]
    bare = host[4:] if host.startswith("www.") else host
    if host in BRAND_HOST_MAP:
        return BRAND_HOST_MAP[host]
    if bare in BRAND_HOST_MAP:
        return BRAND_HOST_MAP[bare]
    for keyword, brand in BRAND_KEYWORDS.items():
        if keyword in bare:
            return brand
    return fallback or ""


def parse_days_arg(value: Optional[str], default: int = 30) -> int:
    try:
        days = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(days, 365))


def aggregate_ads(days: int) -> List[Dict[str, Any]]:
    cutoff = utc_now() - timedelta(days=days - 1)
    pipeline = [
        {"$match": {"detected_at": {"$gte": cutoff}}},
        {"$sort": {"detected_at": -1}},
        {
            "$group": {
                "_id": {
                    "ad_id": {"$ifNull": ["$ad_id", ""]},
                    "site": {"$ifNull": ["$site", ""]},
                    "url": {"$ifNull": ["$url", ""]},
                },
                "first_seen": {"$last": "$detected_at"},
                "last_seen": {"$first": "$detected_at"},
                "times_seen": {"$sum": 1},
                "day_tokens": {
                    "$addToSet": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$detected_at",
                            "timezone": "UTC",
                        }
                    }
                },
                "latest_status": {"$first": "$status"},
                "latest_src": {"$first": "$src_open"},
                "latest_screenshot": {"$first": "$screenshot"},
                "latest_url": {"$first": "$url"},
                "latest_brand": {"$first": "$brand"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "group": "$_id",
                "first_seen": "$first_seen",
                "last_seen": "$last_seen",
                "times_seen": "$times_seen",
                "day_tokens": "$day_tokens",
                "status": "$latest_status",
                "src_open": "$latest_src",
                "screenshot": "$latest_screenshot",
                "url": "$latest_url",
                "brand": "$latest_brand",
            }
        },
        {"$sort": {"last_seen": -1, "times_seen": -1}},
    ]
    try:
        docs = list(ads_collection.aggregate(pipeline, allowDiskUse=True))
    except PyMongoError as exc:
        logging.error("Failed to aggregate ads: %s", exc, exc_info=True)
        raise

    today = utc_now().date().isoformat()
    results: List[Dict[str, Any]] = []
    for doc in docs:
        group = doc.get("group") or {}
        ad_id = (group.get("ad_id") or "").strip()
        site = (group.get("site") or "").strip()
        landing_url = doc.get("url") or group.get("url") or ""
        last_seen_dt = doc.get("last_seen")
        first_seen_dt = doc.get("first_seen")
        if not isinstance(last_seen_dt, datetime):
            continue
        if not isinstance(first_seen_dt, datetime):
            first_seen_dt = last_seen_dt
        last_seen_date = iso_date(last_seen_dt)
        first_seen_date = iso_date(first_seen_dt)
        day_tokens = doc.get("day_tokens") or []
        days_seen = len(day_tokens) or 1
        times_seen = int(doc.get("times_seen") or 0)
        brand = doc.get("brand") or infer_brand(landing_url)
        status = (doc.get("status") or "").lower()
        if last_seen_date == today:
            status = "active"
        elif status not in {"active", "inactive"}:
            status = "inactive"

        results.append(
            {
                "ad_id": ad_id,
                "site": site,
                "brand": brand,
                "first_seen_date": first_seen_date,
                "last_seen_date": last_seen_date,
                "days_seen": days_seen,
                "times_seen": times_seen,
                "status": status,
                "src_open": doc.get("src_open") or "",
                "landing_open": landing_url,
                "screenshot": doc.get("screenshot") or "",
            }
        )
    return results


def build_tsv(rows: Iterable[Dict[str, Any]]) -> io.BytesIO:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=REPORT_HEADERS, delimiter="\t")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in REPORT_HEADERS})
    data = buf.getvalue().encode("utf-8")
    stream = io.BytesIO(data)
    stream.seek(0)
    return stream


def build_xlsx(rows: List[Dict[str, Any]]) -> Tuple[io.BytesIO, str, str]:
    try:
        import pandas as pd
    except ImportError:
        logging.warning("pandas not installed, falling back to TSV for XLSX export")
        stream = build_tsv(rows)
        return stream, "text/tab-separated-values; charset=utf-8", ".tsv"
    df = pd.DataFrame(rows, columns=REPORT_HEADERS)
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="ads", index=False)
    bio.seek(0)
    mimetype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return bio, mimetype, ".xlsx"


def wipe_directory_contents(path: Path) -> int:
    if not path.exists():
        return 0
    removed = 0
    for child in path.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            removed += 1
        except Exception as exc:
            logging.warning("Failed to remove %s: %s", child, exc)
    path.mkdir(parents=True, exist_ok=True)
    return removed


def scraper_is_running() -> bool:
    thread = scrape_thread
    return thread is not None and thread.is_alive()


def _run_scraper_subprocess() -> None:
    global scrape_thread
    env = os.environ.copy()
    env.setdefault("SKIP_API", "0")
    env.setdefault("INGEST_BASE", os.getenv("INGEST_BASE", f"http://127.0.0.1:{PORT}"))
    cmd = [sys.executable, "run.py"]
    logging.info("Starting scraper subprocess: %s", " ".join(cmd))
    try:
        subprocess.run(
            cmd,
            cwd=str(SCRAPER_DIR),
            env=env,
            check=True,
            timeout=SCRAPER_TIMEOUT,
        )
        logging.info("Scraper subprocess completed successfully")
    except subprocess.CalledProcessError as exc:
        logging.error("Scraper failed (code=%s): %s", exc.returncode, exc)
        with scrape_lock:
            scrape_state["error"] = f"Scraper failed (code={exc.returncode})"
    except subprocess.TimeoutExpired:
        logging.error("Scraper timed out after %s seconds", SCRAPER_TIMEOUT)
        with scrape_lock:
            scrape_state["error"] = f"Scraper timed out after {SCRAPER_TIMEOUT}s"
    except Exception as exc:
        logging.exception("Unexpected scraper failure: %s", exc)
        with scrape_lock:
            scrape_state["error"] = f"Unexpected error: {exc}"
    finally:
        with scrape_lock:
            scrape_state["running"] = False
            scrape_state["finished_at"] = utc_now()
            scrape_thread = None


def start_scraper_job() -> bool:
    global scrape_thread
    with scrape_lock:
        if scraper_is_running():
            return False
        scrape_state["running"] = True
        scrape_state["started_at"] = utc_now()
        scrape_state["finished_at"] = None
        scrape_state["error"] = None
        scrape_thread = Thread(target=_run_scraper_subprocess, daemon=True, name="scraper-runner")
        scrape_thread.start()
    return True


def chatbot_command_router(message: str) -> Optional[Dict[str, str]]:
    normalized = (message or "").strip().lower()
    if not normalized:
        return None

    report_keywords = ["ads", "dashboard", "report", "статист", "тайлан"]
    if any(keyword in normalized for keyword in report_keywords):
        msg = (
            "Ads тайланг дараах холбоосуудаас татаж болно:\n"
            "- TSV: /report/ads.tsv\n"
            "- XLSX: /report/ads.xlsx\n"
            "Realtime хүснэгтийг /admin хуудаснаас шалгана уу."
        )
        return {"response": msg}

    scrape_keywords = ["scrape", "шинэчлэх", "шинэчил", "reset ads", "зар тат"]
    if any(keyword in normalized for keyword in scrape_keywords):
        started = start_scraper_job()
        if started:
            note = "Шинэ таталт эхэллээ. /admin хуудаснаас явцыг шалгана уу."
        else:
            note = "Скрепер аль хэдийн ажиллаж байна. Дуусмагц /admin дээр шинэчигдэнэ."
        return {"response": note}

    return None


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/admin")
def admin_ads() -> str:
    return render_template("admin_ads.html")


@app.route("/chatbot", methods=["POST"])
def chatbot() -> Any:
    data = request.get_json(silent=True) or request.form
    message = (data.get("question") or data.get("message") or "").strip()
    if not message:
        return jsonify({"response": "Асуулт хоосон байна, дахин оролдоно уу."})
    command = chatbot_command_router(message)
    if command:
        return jsonify(command)
    return jsonify({"response": processor.chatbot_response(message)})


@app.get("/ads/api/summary")
def api_ads_summary() -> Any:
    days = parse_days_arg(request.args.get("days"), default=30)
    try:
        rows = aggregate_ads(days)
    except PyMongoError:
        return jsonify({"ok": False, "error": "database_error"}), 500
    return jsonify({"ok": True, "items": rows, "meta": {"days": days, "count": len(rows)}})


@app.post("/admin/scrape-now")
def api_scrape_now() -> Any:
    started = start_scraper_job()
    status = {
        "running": scraper_is_running(),
        "started": scrape_state.get("started_at").isoformat() if scrape_state.get("started_at") else None,
        "finished": scrape_state.get("finished_at").isoformat() if scrape_state.get("finished_at") else None,
        "last_error": scrape_state.get("error"),
    }
    if started:
        logging.info("Scraper job queued by admin request")
    return jsonify({"ok": True, "started": started, "status": status})


@app.post("/ads/api/scrape")
def api_scrape_alias() -> Any:
    return api_scrape_now()


def _require_ingest_token() -> None:
    if not INGEST_TOKEN:
        return
    provided = request.headers.get("X-INGEST-TOKEN", "")
    if provided != INGEST_TOKEN:
        abort(401)


@app.post("/ads/api/upload")
def api_upload() -> Any:
    _require_ingest_token()
    file: Optional[FileStorage] = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "missing_file"}), 400

    today_dir = UPLOAD_ROOT / utc_now().strftime("%Y-%m-%d")
    today_dir.mkdir(parents=True, exist_ok=True)

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"ok": False, "error": "invalid_filename"}), 400

    target = today_dir / filename
    if target.exists():
        stem = target.stem
        suffix = target.suffix
        target = today_dir / f"{stem}_{int(utc_now().timestamp())}{suffix}"

    file.save(target)
    rel_path = target.relative_to(STATIC_DIR)
    url = f"/static/{rel_path.as_posix()}"
    logging.info("Uploaded ad asset saved to %s", target)
    return jsonify({"ok": True, "url": url})


@app.post("/ads/api/ingest")
def api_ingest() -> Any:
    _require_ingest_token()
    payload = request.get_json(silent=True) or {}
    ad_id = (payload.get("ad_id") or "").strip()
    site = (payload.get("site") or "").strip()
    status = (payload.get("status") or "active").strip() or "active"
    if not ad_id:
        return jsonify({"ok": False, "error": "ad_id required"}), 400
    if not site:
        return jsonify({"ok": False, "error": "site required"}), 400

    landing_url = (payload.get("url") or payload.get("landing_url") or "").strip()
    src_open = (payload.get("src_open") or payload.get("src") or "").strip()
    screenshot = (payload.get("screenshot") or "").strip()
    detected_at = utc_now()

    doc = {
        "ad_id": ad_id,
        "site": site,
        "status": status.lower(),
        "url": landing_url,
        "src_open": src_open,
        "screenshot": screenshot,
        "detected_at": detected_at,
        "brand": payload.get("brand") or infer_brand(landing_url),
        "ingest_payload": payload,
    }

    try:
        ads_collection.insert_one(doc)
        logging.info("Ingested ad %s from %s (%s)", ad_id, site, landing_url)
    except PyMongoError as exc:
        logging.error("Failed to insert ad ingest: %s", exc, exc_info=True)
        return jsonify({"ok": False, "error": "database_error"}), 500

    return jsonify({"ok": True, "detected_at": detected_at.isoformat()})


@app.get("/report/ads.tsv")
def report_ads_tsv() -> Any:
    days = parse_days_arg(request.args.get("days"), default=30)
    try:
        rows = aggregate_ads(days)
    except PyMongoError:
        return jsonify({"ok": False, "error": "database_error"}), 500
    stream = build_tsv(rows)
    filename = f"ads_{utc_now().date().isoformat()}_{days}d.tsv"
    return send_file(
        stream,
        as_attachment=True,
        download_name=filename,
        mimetype="text/tab-separated-values; charset=utf-8",
    )


@app.get("/report/ads.xlsx")
def report_ads_xlsx() -> Any:
    days = parse_days_arg(request.args.get("days"), default=30)
    try:
        rows = aggregate_ads(days)
    except PyMongoError:
        return jsonify({"ok": False, "error": "database_error"}), 500
    stream, mimetype, suffix = build_xlsx(rows)
    filename = f"ads_{utc_now().date().isoformat()}_{days}d{suffix}"
    return send_file(stream, as_attachment=True, download_name=filename, mimetype=mimetype)


@app.get("/tools/reset")
def tools_reset() -> Any:
    if not RESET_TOKEN:
        abort(404)
    token = request.args.get("token", "")
    if token != RESET_TOKEN:
        abort(403)

    deleted = ads_collection.delete_many({}).deleted_count
    removed_static = wipe_directory_contents(UPLOAD_ROOT)
    removed_shots = wipe_directory_contents(BANNER_SCREENSHOT_DIR)
    logging.info("Test data reset: docs=%s static=%s shots=%s", deleted, removed_static, removed_shots)
    return jsonify(
        {
            "ok": True,
            "deleted_documents": deleted,
            "removed_static_files": removed_static,
            "removed_screenshots": removed_shots,
        }
    )


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=PORT, debug=debug)
