# app.py — Unitel AI Hub (Mongo + Scraper dashboard + Background "Scrape Now")
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
from collections import deque

from dotenv import load_dotenv, dotenv_values
from flask import (
    Flask, Blueprint, abort, jsonify, render_template, request, send_file,
    send_from_directory
)
from flask_cors import CORS
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

import processor  # ← sentiment/чатботын логик энд

# --------------------------- Boot & Config ---------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("app")

ROOT_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = ROOT_DIR / "templates"
STATIC_DIR = ROOT_DIR / "static"

def detect_scraper_dir(root: Path) -> Path:
    override = (os.getenv("SCRAPER_DIR_OVERRIDE") or os.getenv("SCRAPER_DIR") or "").strip()
    if override:
        p = Path(override)
        return (p if p.is_absolute() else (root / p)).resolve()
    for cand in (root / "Scraper", root / "scraper"):
        if (cand / "run.py").exists():
            return cand.resolve()
    return root.resolve()

SCRAPER_DIR = detect_scraper_dir(ROOT_DIR)
SCRAPER_EXPORT_DIR = SCRAPER_DIR / "_export"
BANNER_SCREENSHOT_DIR = (Path(os.getenv("OUT_DIR") or (SCRAPER_DIR / "banner_screenshots"))).resolve()
TSV_PATH = Path(os.getenv("TSV_PATH") or (SCRAPER_DIR / "banner_tracking_combined.tsv")).resolve()
UPLOAD_ROOT = STATIC_DIR / "ads"
LOG_DIR = SCRAPER_DIR / "_logs"
for d in (STATIC_DIR, UPLOAD_ROOT, BANNER_SCREENSHOT_DIR, SCRAPER_EXPORT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
app.config["JSON_SORT_KEYS"] = False
CORS(app)

PORT = int(os.getenv("PORT", "8888"))
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "unitel_aihub")
ADS_COLLECTION_NAME = os.getenv("ADS_COLLECTION", "ads_events")
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "")
RESET_TOKEN = os.getenv("RESET_TOKEN", "")
SCRAPER_TIMEOUT = int(os.getenv("SCRAPER_TIMEOUT", "1800"))
SUMMARIZE_TIMEOUT = int(os.getenv("SUMMARIZE_TIMEOUT", "300"))

def get_scraper_python() -> str:
    env_py = os.getenv("SCRAPER_PY", "").strip()
    if env_py and Path(env_py).exists():
        return env_py
    candidates = [
        ROOT_DIR / ".venv" / "Scripts" / "python.exe",
        SCRAPER_DIR / ".venv" / "Scripts" / "python.exe",
        ROOT_DIR / "venv" / "Scripts" / "python.exe",
        SCRAPER_DIR / "venv" / "Scripts" / "python.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return sys.executable

# --------------------------- Mongo ---------------------------
mongo_client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
try:
    mongo_client.admin.command("ping")
    log.info("Connected to MongoDB at %s", MONGO_URL)
except Exception as exc:
    log.exception("MongoDB connection failed: %s", exc)
    raise

db = mongo_client[DB_NAME]
ads_collection: Collection = db[ADS_COLLECTION_NAME]
ads_collection.create_index(
    [("ad_id", ASCENDING), ("site", ASCENDING), ("url", ASCENDING), ("detected_at", DESCENDING)],
    name="ads_lookup",
    background=True,
)
ads_collection.create_index([("detected_at", DESCENDING)], name="detected_at_desc", background=True)

REPORT_HEADERS = [
    "ad_id","site","brand","first_seen_date","last_seen_date","days_seen","times_seen",
    "status","src_open","landing_open","screenshot",
]

BRAND_HOST_MAP: Dict[str, str] = {
    "unitel.mn":"Unitel","www.unitel.mn":"Unitel","shop.unitel.mn":"Unitel",
    "skytel.mn":"Skytel","www.skytel.mn":"Skytel",
    "mobicom.mn":"Mobicom","www.mobicom.mn":"Mobicom",
    "gmobile.mn":"G-Mobile","www.gmobile.mn":"G-Mobile",
    "ardcredit.mn":"ARD Credit","www.ardcredit.mn":"ARD Credit",
    "ard.mn":"ARD Bank","www.ard.mn":"ARD Bank",
    "khanbank.com":"Khan Bank","www.khanbank.com":"Khan Bank",
    "golomtbank.com":"Golomt Bank","www.golomtbank.com":"Golomt Bank",
    "bogdbank.com":"Bogd Bank","www.bogdbank.com":"Bogd Bank",
}
BRAND_KEYWORDS: Dict[str, str] = {
    "unitel":"Unitel","skytel":"Skytel","mobicom":"Mobicom","gmobile":"G-Mobile",
    "ardcredit":"ARD Credit","ardbank":"ARD Bank","ard":"ARD",
    "khanbank":"Khan Bank","golomt":"Golomt Bank","bogd":"Bogd Bank",
    "tavanbogd":"Таван Богд","xacbank":"Хас Банк",
}

# --------------------------- State ---------------------------
scrape_lock = Lock()
scrape_thread: Optional[Thread] = None
scrape_state: Dict[str, Any] = {"running": False, "started_at": None, "finished_at": None, "error": None}
LAST_LOG = deque(maxlen=400)

# --------------------------- Utils ---------------------------
def utc_now() -> datetime: return datetime.now(timezone.utc)
def iso_date(dt: datetime) -> str: return dt.astimezone(timezone.utc).date().isoformat()
def host_from_url(url: str) -> str:
    if not url: return ""
    try: return urlparse(url).netloc.lower()
    except Exception: return ""
def infer_brand(url: str, fallback: Optional[str] = None) -> str:
    host = host_from_url(url)
    if not host: return fallback or ""
    host = host.split(":")[0]
    bare = host[4:] if host.startswith("www.") else host
    if host in BRAND_HOST_MAP: return BRAND_HOST_MAP[host]
    if bare in BRAND_HOST_MAP: return BRAND_HOST_MAP[bare]
    for kw, brand in BRAND_KEYWORDS.items():
        if kw in bare: return brand
    return fallback or ""
def parse_days_arg(value: Optional[str], default: int = 30) -> int:
    try: days = int(value)
    except (TypeError, ValueError): return default
    return max(1, min(days, 365))
def _logl(s: str):
    s = s.rstrip("\n")
    LAST_LOG.append(s)
    log.info("[SCRAPER] %s", s)

# --------------------------- Aggregation (Mongo API) ---------------------------
def aggregate_ads(days: int) -> List[Dict[str, Any]]:
    cutoff = utc_now() - timedelta(days=days - 1)
    pipeline = [
        {"$match": {"detected_at": {"$gte": cutoff}}},
        {"$sort": {"detected_at": -1}},
        {"$group": {
            "_id": {"ad_id":{"$ifNull":["$ad_id",""]},"site":{"$ifNull":["$site",""]},"url":{"$ifNull":["$url",""]}},
            "first_seen":{"$last":"$detected_at"},"last_seen":{"$first":"$detected_at"},"times_seen":{"$sum":1},
            "day_tokens":{"$addToSet":{"$dateToString":{"format":"%Y-%m-%d","date":"$detected_at","timezone":"UTC"}}},
            "latest_status":{"$first":"$status"},"latest_src":{"$first":"$src_open"},
            "latest_screenshot":{"$first":"$screenshot"},"latest_url":{"$first":"$url"},
            "latest_brand":{"$first":"$brand"},
        }},
        {"$project":{
            "_id":0,"group":"$_id","first_seen":"$first_seen","last_seen":"$last_seen","times_seen":"$times_seen",
            "day_tokens":"$day_tokens","status":"$latest_status","src_open":"$latest_src",
            "screenshot":"$latest_screenshot","url":"$latest_url","brand":"$latest_brand",
        }},
        {"$sort": {"last_seen": -1, "times_seen": -1}},
    ]
    docs = list(ads_collection.aggregate(pipeline, allowDiskUse=True))
    today = utc_now().date().isoformat()
    out: List[Dict[str, Any]] = []
    for doc in docs:
        group = doc.get("group") or {}
        ad_id = (group.get("ad_id") or "").strip()
        site = (group.get("site") or "").strip()
        landing_url = doc.get("url") or group.get("url") or ""
        last_seen_dt = doc.get("last_seen")
        first_seen_dt = doc.get("first_seen") or last_seen_dt
        if not isinstance(last_seen_dt, datetime): continue
        last_seen_date = iso_date(last_seen_dt)
        first_seen_date = iso_date(first_seen_dt)
        days_seen = len(doc.get("day_tokens") or []) or 1
        times_seen = int(doc.get("times_seen") or 0)
        brand = doc.get("brand") or infer_brand(landing_url)
        status = (doc.get("status") or "").lower()
        status = "active" if last_seen_date == today else ("inactive" if status not in {"active","inactive"} else status)
        out.append({
            "ad_id": ad_id, "site": site, "brand": brand,
            "first_seen_date": first_seen_date, "last_seen_date": last_seen_date,
            "days_seen": days_seen, "times_seen": times_seen, "status": status,
            "src_open": doc.get("src_open") or "", "landing_open": landing_url, "screenshot": doc.get("screenshot") or "",
        })
    return out

def build_tsv(rows: Iterable[Dict[str, Any]]) -> io.BytesIO:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=REPORT_HEADERS, delimiter="\t")
    w.writeheader()
    for r in rows: w.writerow({k: r.get(k, "") for k in REPORT_HEADERS})
    bio = io.BytesIO(buf.getvalue().encode("utf-8")); bio.seek(0); return bio

def build_xlsx(rows: List[Dict[str, Any]]) -> Tuple[io.BytesIO, str, str]:
    try:
        import pandas as pd
    except ImportError:
        log.warning("pandas not installed, falling back to TSV for XLSX export")
        s = build_tsv(rows); return s, "text/tab-separated-values; charset=utf-8", ".tsv"
    df = pd.DataFrame(rows, columns=REPORT_HEADERS)
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="ads", index=False)
    bio.seek(0)
    return bio, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"

def wipe_directory_contents(path: Path) -> int:
    if not path.exists(): return 0
    removed = 0
    for ch in path.iterdir():
        try:
            shutil.rmtree(ch) if ch.is_dir() else ch.unlink()
            removed += 1
        except Exception as exc:
            log.warning("Failed to remove %s: %s", ch, exc)
    path.mkdir(parents=True, exist_ok=True)
    return removed

# --------------------------- Scraper runner ---------------------------
def scraper_is_running() -> bool:
    th = scrape_thread
    return th is not None and th.is_alive()

def _run_scraper_subprocess() -> None:
    """run.py → (optional) summarize.py – child process, env merged."""
    global scrape_thread
    py = get_scraper_python()
    env = os.environ.copy()
    try:
        env.update(dotenv_values(SCRAPER_DIR / ".env"))
    except Exception:
        pass
    env.setdefault("SKIP_API", "0")
    env.setdefault("INGEST_BASE", os.getenv("INGEST_BASE", f"http://127.0.0.1:{PORT}"))
    if INGEST_TOKEN: env["INGEST_TOKEN"] = INGEST_TOKEN
    env["OUT_DIR"] = str(BANNER_SCREENSHOT_DIR)
    env["PYTHONUNBUFFERED"] = "1"

    try:
        cmd = [py, "run.py"] + (os.getenv("SCRAPER_ARGS", "") or "").split()
        _logl(f"launching: {' '.join(cmd)} | cwd={SCRAPER_DIR}")
        subprocess.run(cmd, cwd=str(SCRAPER_DIR), env=env, check=True, timeout=SCRAPER_TIMEOUT)
        _logl("run.py finished OK")

        if (SCRAPER_DIR / "summarize.py").exists():
            cmd2 = [py, "summarize.py"]
            _logl(f"launching: {' '.join(cmd2)} | cwd={SCRAPER_DIR}")
            subprocess.run(cmd2, cwd=str(SCRAPER_DIR), env=env, check=True, timeout=SUMMARIZE_TIMEOUT)
            _logl("summarize.py finished OK")

    except subprocess.CalledProcessError as exc:
        log.error("Scraper failed (code=%s): %s", exc.returncode, exc)
        with scrape_lock: scrape_state["error"] = f"Scraper failed (code={exc.returncode})"
    except subprocess.TimeoutExpired:
        log.error("Scraper timed out after %s seconds", SCRAPER_TIMEOUT)
        with scrape_lock: scrape_state["error"] = f"Scraper timed out after {SCRAPER_TIMEOUT}s"
    except Exception as exc:
        log.exception("Unexpected scraper failure: %s", exc)
        with scrape_lock: scrape_state["error"] = f"Unexpected error: {exc}"
    finally:
        with scrape_lock:
            scrape_state["running"] = False
            scrape_state["finished_at"] = utc_now()
            scrape_thread = None
        _logl("THREAD END")

def start_scraper_job() -> bool:
    global scrape_thread
    with scrape_lock:
        if scraper_is_running():
            return False
        scrape_state.update({"running": True, "started_at": utc_now(), "finished_at": None, "error": None})
        scrape_thread = Thread(target=_run_scraper_subprocess, daemon=True, name="scraper-runner")
        scrape_thread.start()
    return True

# --------------------------- Chat hook ---------------------------
def chatbot_command_router(message: str) -> Optional[Dict[str, str]]:
    normalized = (message or "").strip().lower()
    if not normalized: return None
    if any(k in normalized for k in ["ads","dashboard","report","статист","тайлан"]):
        msg = ("Ads тайлан:\n- TSV: /scraper/download/tsv\n- XLSX: /scraper/download/xlsx\nRealtime хүснэгт: /scraper/")
        return {"response": msg}
    if any(k in normalized for k in ["scrape","шинэчлэх","шинэчил","reset ads","зар тат"]):
        started = start_scraper_job()
        note = "Шинэ таталт эхэллээ. /scraper/ дээр явцыг харна." if started else "Скрепер аль хэдийн ажиллаж байна."
        return {"response": note}
    return None

# --------------------------- Root page ---------------------------
@app.get("/")
def index() -> str:
    return render_template("index.html")

# --------------------------- Blueprint: /scraper ---------------------------
scraper_bp = Blueprint("scraper", __name__, url_prefix="/scraper")

@scraper_bp.route("/")
def report():
    rows: List[Dict[str, str]] = []
    if TSV_PATH.exists():
        try:
            with TSV_PATH.open("r", encoding="utf-8", newline="") as fh:
                rd = csv.DictReader(fh, delimiter="\t")
                today = datetime.now(timezone.utc).date().isoformat()
                for r in rd:
                    # --- normalize: screenshot path → /shots/.. ---
                    r["screenshot_file"] = r.get("screenshot_file") or r.get("screenshot_path") or r.get("screenshot") or ""
                    if r["screenshot_file"]:
                        s = r["screenshot_file"].replace("\\", "/")
                        try:
                            ap = Path(s); ap = ap if ap.is_absolute() else Path("/" + s.lstrip("/"))
                            rel = ap.resolve().relative_to(BANNER_SCREENSHOT_DIR)
                            r["screenshot_file"] = f"/shots/{rel.as_posix()}"
                        except Exception:
                            key = "/banner_screenshots/"; i = s.lower().find(key)
                            r["screenshot_file"] = f"/shots/{s[i+len(key):]}" if i != -1 else s

                    # --- normalize: landing URL field name(s) ---
                    if not r.get("landing_open"):
                        r["landing_open"] = r.get("landing_url") or r.get("url") or ""

                    # --- normalize: brand / score / dates ---
                    if not (r.get("brand") or "").strip():
                        r["brand"] = infer_brand(r.get("landing_open") or "")
                    r["ad_score"] = r.get("ad_score") or r.get("score") or r.get("оноо") or r.get("rank") or ""
                    r["last_seen_date"] = r.get("last_seen_date") or r.get("last_seen") or r.get("last_seen_at") or r.get("updated_date") or r.get("last_date") or r.get("last_seen_date")
                    r["first_seen_date"] = r.get("first_seen_date") or r.get("first_seen") or r.get("created_date") or r.get("first_date") or r.get("first_seen_date")

                    # --- normalize: status → ИДЭВХТЭЙ / ДУУССАН ---
                    raw = (r.get("status") or r.get("state") or r.get("төлөв") or "").strip().lower()
                    if not raw:
                        status_mn = "ИДЭВХТЭЙ" if r.get("last_seen_date") == today else "ДУУССАН"
                    else:
                        if any(k in raw for k in ("active", "идэвх", "running", "live")):
                            status_mn = "ИДЭВХТЭЙ"
                        elif any(k in raw for k in ("inactive", "дуус", "finished", "stopped", "expired")):
                            status_mn = "ДУУССАН"
                        else:
                            status_mn = "ИДЭВХТЭЙ" if r.get("last_seen_date") == today else "ДУУССАН"
                    r["status"] = status_mn

                    rows.append(r)
        except Exception as exc:
            log.warning("TSV read error %s: %s", TSV_PATH, exc)

    return render_template(
        "scraper.html",
        rows=rows,
        tsv_exists=TSV_PATH.exists(),
        xlsx_exists=(SCRAPER_EXPORT_DIR / "summary.xlsx").exists(),
    )

@scraper_bp.get("/status")
def scraper_status_bp():
    return jsonify({
        "running": scraper_is_running(),
        "started": scrape_state.get("started_at").isoformat() if scrape_state.get("started_at") else None,
        "finished": scrape_state.get("finished_at").isoformat() if scrape_state.get("finished_at") else None,
        "last_error": scrape_state.get("error"),
        "log_tail": list(LAST_LOG)[-12:],
    })

@scraper_bp.post("/scrape-now")
def scrape_now_bp():
    started = start_scraper_job()
    status = {
        "running": scraper_is_running(),
        "started": scrape_state.get("started_at").isoformat() if scrape_state.get("started_at") else None,
        "finished": scrape_state.get("finished_at").isoformat() if scrape_state.get("finished_at") else None,
        "last_error": scrape_state.get("error"),
    }
    log.info("UI trigger /scraper/scrape-now -> started=%s  cwd=%s  py=%s", started, SCRAPER_DIR, get_scraper_python())
    return jsonify({"ok": True, "started": started, "status": status})

@scraper_bp.get("/download/tsv")
def download_tsv():
    if not TSV_PATH.exists():
        abort(404)
    return send_file(TSV_PATH, as_attachment=True, download_name="banner_tracking_combined.tsv")

@scraper_bp.get("/download/xlsx")
def download_xlsx():
    x = SCRAPER_EXPORT_DIR / "summary.xlsx"
    if not x.exists():
        abort(404)
    return send_from_directory(SCRAPER_EXPORT_DIR, "summary.xlsx", as_attachment=True)

app.register_blueprint(scraper_bp)

# --------------------------- Chatbot API (Sentiment + Files) ---------------------------
@app.post("/chatbot")
def chatbot() -> Any:
    # 1) multipart/form-data (текст + файлууд)
    if request.content_type and request.content_type.startswith("multipart/form-data"):
        message = (request.form.get("message") or request.form.get("question") or "").strip()
        files: List[FileStorage] = request.files.getlist("files")
        tmp_dir = (Path(app.static_folder) / "tmp").resolve()
        tmp_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: List[str] = []
        for f in files:
            if not f or not f.filename:
                continue
            p = tmp_dir / secure_filename(f.filename)
            f.save(p)
            saved_paths.append(str(p))
        reply = processor.chatbot_response(message, files=saved_paths)
        return jsonify({"response": reply})

    # 2) JSON only
    data = request.get_json(silent=True) or {}
    message = (data.get("question") or data.get("message") or "").strip()
    if not message:
        return jsonify({"response": "Асуулт хоосон байна, дахин оролдоно уу."})
    command = chatbot_command_router(message)
    if command:
        return jsonify(command)
    return jsonify({"response": processor.chatbot_response(message)})

# --------------------------- Ads summary API ---------------------------
@app.get("/ads/api/summary")
def api_ads_summary() -> Any:
    days = parse_days_arg(request.args.get("days"), default=30)
    try:
        rows = aggregate_ads(days)
    except PyMongoError:
        return jsonify({"ok": False, "error": "database_error"}), 500
    return jsonify({"ok": True, "items": rows, "meta": {"days": days, "count": len(rows)}})

# ---- Ingest / Upload ----
def _require_ingest_token() -> None:
    if not INGEST_TOKEN: return
    provided = request.headers.get("X-INGEST-TOKEN", "")
    if provided != INGEST_TOKEN: abort(401)

@app.post("/ads/api/upload")
def api_upload() -> Any:
    _require_ingest_token()
    file: Optional[FileStorage] = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "missing_file"}), 400
    today_dir = UPLOAD_ROOT / utc_now().strftime("%Y-%m-%d")
    today_dir.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(file.filename)
    if not filename: return jsonify({"ok": False, "error": "invalid_filename"}), 400
    target = today_dir / filename
    if target.exists():
        target = today_dir / f"{target.stem}_{int(utc_now().timestamp())}{target.suffix}"
    file.save(target)
    rel = target.relative_to(STATIC_DIR)
    url = f"/static/{rel.as_posix()}"
    log.info("Uploaded ad asset saved to %s", target)
    return jsonify({"ok": True, "url": url})

@app.post("/ads/api/ingest")
def api_ingest() -> Any:
    _require_ingest_token()
    payload = request.get_json(silent=True) or {}
    ad_id = (payload.get("ad_id") or "").strip()
    site = (payload.get("site") or "").strip()
    status = (payload.get("status") or "active").strip() or "active"
    if not ad_id: return jsonify({"ok": False, "error": "ad_id required"}), 400
    if not site: return jsonify({"ok": False, "error": "site required"}), 400
    landing_url = (payload.get("url") or payload.get("landing_url") or "").strip()
    src_open = (payload.get("src_open") or payload.get("src") or "").strip()
    screenshot = (payload.get("screenshot") or "").strip()
    detected_at = utc_now()
    doc = {
        "ad_id": ad_id, "site": site, "status": status.lower(), "url": landing_url,
        "src_open": src_open, "screenshot": screenshot, "detected_at": detected_at,
        "brand": payload.get("brand") or infer_brand(landing_url), "ingest_payload": payload,
    }
    try:
        ads_collection.insert_one(doc)
        log.info("Ingested ad %s from %s (%s)", ad_id, site, landing_url)
    except PyMongoError as exc:
        log.error("Failed to insert ad ingest: %s", exc, exc_info=True)
        return jsonify({"ok": False, "error": "database_error"}), 500
    return jsonify({"ok": True, "detected_at": detected_at.isoformat()})

# ---- Serve screenshots ----
@app.get("/shots/<path:relpath>")
def serve_shot(relpath: str):
    p = (BANNER_SCREENSHOT_DIR / relpath).resolve()
    if not str(p).startswith(str(BANNER_SCREENSHOT_DIR)) or not p.exists():
        abort(404)
    return send_file(p)

# ---- Tools ----
@app.get("/tools/reset")
def tools_reset() -> Any:
    if not RESET_TOKEN: abort(404)
    if request.args.get("token","") != RESET_TOKEN: abort(403)
    deleted = ads_collection.delete_many({}).deleted_count
    removed_static = wipe_directory_contents(UPLOAD_ROOT)
    removed_shots = wipe_directory_contents(BANNER_SCREENSHOT_DIR)
    log.info("Test data reset: docs=%s static=%s shots=%s", deleted, removed_static, removed_shots)
    return jsonify({"ok": True,"deleted_documents":deleted,"removed_static_files":removed_static,"removed_screenshots":removed_shots})

# ---- Main ----
if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=PORT, debug=debug, use_reloader=False, threaded=True)
