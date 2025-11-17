# app.py — Unitel AI Hub (Auth + Scraper + Ads ingest + Chatbot)
from __future__ import annotations

import csv
import io
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from threading import Lock, Thread
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse
from collections import deque
import re
from flask import send_from_directory
from pathlib import Path


from dotenv import load_dotenv, dotenv_values
from flask import (
    Flask, Blueprint, abort, jsonify, render_template, request, send_file,
    send_from_directory, redirect, url_for
)
from flask_cors import CORS
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId



# ---- Auth (Flask-Login)
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    current_user, login_required
)
from processor import process_query


# --------------------------- Boot & Config ---------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("app")

# ✅ 1. ЗАСВАР: ROOT_DIR, BASE_DIR, DOWNLOADS_DIR-г энд зөв дарааллаар тодорхойлов.
ROOT_DIR = Path(__file__).resolve().parent
BASE_DIR = ROOT_DIR
DOWNLOADS_DIR = os.path.join(BASE_DIR, "data", "downloads")

TEMPLATES_DIR = ROOT_DIR / "templates"
STATIC_DIR = ROOT_DIR / "static"

# ⚠️ АСУУДАЛ 3-ИЙН ЗАСВАРЫН НЭГ ХЭСЭГ (Data/Tmp тодорхойлох)
DATA_DIR = ROOT_DIR / "data"
TMP_DIR = STATIC_DIR / "tmp"


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

# ⚠️ АСУУДАЛ 3-ИЙН ЗАСВАР (chmod 777 -> mode=0o777)
# Шаардлагатай бүх хавтаснуудыг 777 эрхтэйгээр (mode=0o777) үүсгэнэ.
for d in (
    STATIC_DIR, UPLOAD_ROOT, BANNER_SCREENSHOT_DIR, SCRAPER_EXPORT_DIR, LOG_DIR,
    DATA_DIR, Path(DOWNLOADS_DIR), TMP_DIR
):
    d.mkdir(parents=True, exist_ok=True, mode=0o777)

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
app.config["JSON_SORT_KEYS"] = False

# ✅ 2. ЗАСВАР: Давхардсан, login-гүй route-ийг эндээс устгасан.
# @app.route('/downloads/<path:filename>') ... (УСТГАСАН)

# --- Session/Cookie хамгаалалт (prod-д HTTPS үед secure)
SECURE_COOKIES = os.getenv("SESSION_COOKIE_SECURE", "0") == "1"  
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    REMEMBER_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=SECURE_COOKIES,
    REMEMBER_COOKIE_SECURE=SECURE_COOKIES,
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
)

# CORS (шаардлагатай бол ENV-д CORS_ORIGINS-г зориулах)
CORS(app,
     resources={r"/*": {"origins": os.getenv("CORS_ORIGINS", "*").split(",")}},
     supports_credentials=True)

# Secret for sessions (FLASK_SECRET -> SECRET_KEY fallback)
app.secret_key = os.getenv("FLASK_SECRET") or os.getenv("SECRET_KEY") or "dev_secret_change_me"

PORT = int(os.getenv("PORT", "8888"))
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "unitel_aihub")
ADS_COLLECTION_NAME = os.getenv("ADS_COLLECTION", "ads_events")
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "")
RESET_TOKEN = os.getenv("RESET_TOKEN", "")
SCRAPER_TIMEOUT = int(os.getenv("SCRAPER_TIMEOUT", "1800"))
SUMMARIZE_TIMEOUT = int(os.getenv("SUMMARIZE_TIMEOUT", "300"))

# ⚠️ АСУУДАЛ 2-ЫН ЗАСВАР (Windows + Linux venv paths)
def get_scraper_python() -> str:
    env_py = os.getenv("SCRAPER_PY", "").strip()
    if env_py and Path(env_py).exists():
        return env_py
    
    candidates = [
        # Linux/macOS paths
        ROOT_DIR / ".venv" / "bin" / "python",
        SCRAPER_DIR / ".venv" / "bin" / "python",
        ROOT_DIR / "venv" / "bin" / "python",
        SCRAPER_DIR / "venv" / "bin" / "python",
        # Windows paths
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

# ---- Users collection (for auth)
users_collection: Collection = db["users"]
users_collection.create_index([("email", ASCENDING)], unique=True, background=True)
users_collection.create_index([("approved", ASCENDING), ("created_at", DESCENDING)], background=True)

# --------------------------- Auth (Flask-Login) ---------------------------
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ALLOW_DOMAIN -> dynamic email regex
ALLOW_DOMAIN = (os.getenv("ALLOW_DOMAIN") or "@unitel.mn").strip()
_allow = re.escape(ALLOW_DOMAIN.lstrip("@"))  # unitel.mn -> escape
EMAIL_RE = re.compile(rf"^[^@\s]+@{_allow}$", re.IGNORECASE)

class User(UserMixin):
    def __init__(self, doc: Dict[str, Any]):
        self._doc = doc
        self.id = str(doc["_id"])
        self.email = doc["email"]
        self.name = doc.get("name") or self.email.split("@")[0]
        self.role = doc.get("role", "user")
        self.approved = bool(doc.get("approved", False))

    @property
    def is_active(self) -> bool:
        return bool(self.approved)

def _to_user(doc: Optional[Dict[str, Any]]) -> Optional[User]:
    return User(doc) if doc else None

@login_manager.user_loader
def load_user(user_id: str) -> Optional[User]:
    try:
        doc = users_collection.find_one({"_id": ObjectId(user_id)})
    except Exception:
        doc = None
    return _to_user(doc)

def admin_required(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for("login"))
        if getattr(current_user, "role", "user") != "admin":
            abort(403)
        return fn(*args, **kwargs)
    return wrapper

# Login дараах redirect-ийг дотоод URL-д хязгаарлах
def _safe_next_url(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    try:
        u = urlparse(raw)
        if not u.scheme and not u.netloc and raw.startswith("/"):
            return raw
    except Exception:
        pass
    return None

# templates-д домэйн placeholder дамжуулах
@app.context_processor
def inject_globals():
    return {"ALLOW_DOMAIN": ALLOW_DOMAIN}

# --------------------------- Reports / Aggregation ---------------------------
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
    
    # ⚠️ АСУУДАЛ 3-ИЙН ЗАСВАР (Permission)
    path.mkdir(parents=True, exist_ok=True, mode=0o777)
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

# --------------------------- Root page (Protected) ---------------------------
@app.get("/")
@login_required
def index() -> str:
    return render_template("index.html")

# --------------------------- Blueprint: /scraper ---------------------------
scraper_bp = Blueprint("scraper", __name__, url_prefix="/scraper")

@scraper_bp.route("/")
@login_required
def report():
    rows: List[Dict[str, str]] = []
    if TSV_PATH.exists():
        try:
            with TSV_PATH.open("r", encoding="utf-8", newline="") as fh:
                rd = csv.DictReader(fh, delimiter="\t")
                today = datetime.now(timezone.utc).date().isoformat()
                for r in rd:
                    # --- normalize: screenshot path → /shots/.. ---
                    r["screenshot_file"] = (
                        r.get("screenshot_file")
                        or r.get("screenshot_path")
                        or r.get("screenshot")
                        or ""
                    )
                    if r["screenshot_file"]:
                        s = r["screenshot_file"].replace("\\", "/")
                        try:
                            ap = Path(s)
                            ap = ap if ap.is_absolute() else Path("/" + s.lstrip("/"))
                            rel = ap.resolve().relative_to(BANNER_SCREENSHOT_DIR)
                            r["screenshot_file"] = f"/shots/{rel.as_posix()}"
                        except Exception:
                            key = "/banner_screenshots/"
                            i = s.lower().find(key)
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
@login_required
def scraper_status_bp():
    return jsonify({
        "running": scraper_is_running(),
        "started": scrape_state.get("started_at").isoformat() if scrape_state.get("started_at") else None,
        "finished": scrape_state.get("finished_at").isoformat() if scrape_state.get("finished_at") else None,
        "last_error": scrape_state.get("error"),
        "log_tail": list(LAST_LOG)[-12:],
    })

@scraper_bp.post("/scrape-now")
@login_required
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
@login_required
def download_tsv():
    if not TSV_PATH.exists():
        abort(404)
    return send_file(TSV_PATH, as_attachment=True, download_name="banner_tracking_combined.tsv")

@scraper_bp.get("/download/xlsx")
@login_required
def download_xlsx():
    x = SCRAPER_EXPORT_DIR / "summary.xlsx"
    if not x.exists():
        abort(404)
    return send_from_directory(SCRAPER_EXPORT_DIR, "summary.xlsx", as_attachment=True)

app.register_blueprint(scraper_bp)

# --------------------------- Chatbot API (Sentiment + Files) ---------------------------
# --------------------------- Chatbot API (Sentiment + Files) ---------------------------
from flask import make_response
import uuid

@app.post("/chatbot")
@login_required
def chatbot() -> Any:
    """
    Chatbot endpoint — текст болон файл аплоадыг зэрэг дэмжинэ.
    """
    message = ""
    files: List[FileStorage] = []

    # 1️⃣ multipart/form-data (файлтай хүсэлт)
    if request.content_type and request.content_type.startswith("multipart/form-data"):
        message = (request.form.get("message") or request.form.get("question") or "").strip()
        files = request.files.getlist("files")

    # 2️⃣ JSON body (файлгүй, текст хүсэлт)
    elif request.is_json:
        data = request.get_json(silent=True) or {}
        message = (data.get("question") or data.get("message") or "").strip()

    # 3️⃣ Фallback (form-urlencoded)
    else:
        message = (request.form.get("message") or "").strip()

    if not message and not files:
        return jsonify({"response": "Асуулт хоосон байна, дахин оролдоно уу."})

    # --- 4️⃣ Session ID үүсгэх эсвэл cookie-с авах ---
    session_id = request.cookies.get("session_id") or str(uuid.uuid4())

    # --- 5️⃣ Түр хадгалах файл зам ---
    tmp_dir = (Path(app.static_folder) / "tmp").resolve()
    
    # ⚠️ АСУУДАЛ 3-ИЙН ЗАСВАР (Permission)
    tmp_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
    saved_paths: List[str] = []

    for f in files:
        if not f or not f.filename:
            continue
        p = tmp_dir / secure_filename(f.filename)
        f.save(p)
        saved_paths.append(str(p))

    # --- 6️⃣ JSON командыг шалгах ---
    if message:
        try:
            from processor.prompt_builder import chatbot_command_router
            command = chatbot_command_router(message)
            if command:
                return jsonify(command)
        except Exception:
            pass

    # --- 7️⃣ Chatbot процесс дуудах ---
    reply = process_query(message, session_id, files=saved_paths)


    # --- 8️⃣ Session ID-г cookie хэлбэрээр хадгалах ---
    response = make_response(jsonify({"response": reply, "html": True}))
    response.set_cookie("session_id", session_id, max_age=3600 * 24 * 7)  # cookie 7 хоног хадгална
    return response

# ---------------- Downloads (facebook/json/xlsx) ----------------


@app.route("/downloads/<path:filename>")
@login_required
def serve_downloads(filename):
    """
    Татаж авах файлуудыг (JSON/XLSX) project_root/data/downloads/ дотроос serve хийнэ.
    """
    DOWNLOAD_DIR = Path(__file__).resolve().parent / "data" / "downloads"
    file_path = DOWNLOAD_DIR / filename

    if not file_path.exists():
        return "File not found", 404

    return send_from_directory(DOWNLOAD_DIR, filename, as_attachment=True)



# --------------------------- Ads summary API ---------------------------
@app.get("/ads/api/summary")
@login_required
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
    
    # ⚠️ АСУУДАЛ 3-ИЙН ЗАСВАР (Permission)
    today_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
    
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
@login_required
def serve_shot(relpath: str):
    p = (BANNER_SCREENSHOT_DIR / relpath).resolve()
    if not str(p).startswith(str(BANNER_SCREENSHOT_DIR)) or not p.exists():
        abort(404)
    return send_file(p)

# --------------------------- Auth Routes (Login/Register/Admin) ---------------------------
@app.get("/login")
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    return render_template("login.html", email=request.args.get("email",""))

@app.post("/login")
def do_login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    data = request.form or {}
    email = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()
    udoc = users_collection.find_one({"email": email})
    if not udoc or not check_password_hash(udoc.get("password_hash", ""), password):
        return render_template("login.html", error="Имэйл/нууц үг буруу байна.", email=email)
    if not udoc.get("approved", False):
        return render_template("login.html", error="Таны бүртгэл админ зөвшөөрөл хүлээж байна.", email=email)
    login_user(User(udoc))
    users_collection.update_one({"_id": udoc["_id"]}, {"$set": {"last_login": datetime.utcnow()}})
    next_url = _safe_next_url(request.args.get("next"))
    return redirect(next_url or url_for("index"))

@app.get("/register")
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    return render_template("register.html")

@app.post("/register")
def do_register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    data = request.form or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()
    password2 = (data.get("password2") or "").strip()

    if not EMAIL_RE.match(email):
        return render_template("register.html", error=f"{ALLOW_DOMAIN} имэйл шаардлагатай.")

    if len(password) < 8:
        return render_template("register.html", error="Нууц үг хамгийн багадаа 8 тэмдэгт байх ёстой.")
    if password != password2:
        return render_template("register.html", error="Нууц үг хоорондоо таарахгүй байна.")

    if users_collection.find_one({"email": email}):
        return render_template("register.html", error="Энэ имэйлээр бүртгэл аль хэдийн байна.")

    users_collection.insert_one({
        "name": name or email.split("@")[0],
        "email": email,
        "password_hash": generate_password_hash(password),
        "role": "user",
        "approved": False,
        "created_at": datetime.utcnow(),
    })
    return render_template("login.html", info="Бүртгэл амжилттай. Админ зөвшөөрсний дараа нэвтэрнэ үү.")

@app.post("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.get("/admin/users")
@admin_required
def admin_users():
    pend = list(users_collection.find({"approved": False}).sort("created_at", DESCENDING))
    actv = list(users_collection.find({"approved": True}).sort("created_at", DESCENDING))
    return render_template("admin_users.html", pending=pend, active=actv)

# ---- Admin: create user (from Admin panel)
@app.post("/admin/users/create")
@admin_required
def admin_create_user():
    name = (request.form.get("name") or "").strip()
    email = (request.form.get("email") or "").strip().lower()
    role = (request.form.get("role") or "user").strip().lower()
    approved = bool(request.form.get("approved"))

    if not EMAIL_RE.match(email):
        return render_template("admin_users.html",
                               error=f"{ALLOW_DOMAIN} домэйнтэй имэйл шаардлагатай.",
                               pending=list(users_collection.find({"approved": False}).sort("created_at", DESCENDING)),
                               active=list(users_collection.find({"approved": True}).sort("created_at", DESCENDING)))

    if users_collection.find_one({"email": email}):
        return render_template("admin_users.html",
                               error="Энэ имэйл бүртгэлтэй байна.",
                               pending=list(users_collection.find({"approved": False}).sort("created_at", DESCENDING)),
                               active=list(users_collection.find({"approved": True}).sort("created_at", DESCENDING)))

    import secrets
    temp_pw = "U!" + secrets.token_hex(4)

    users_collection.insert_one({
        "name": name or email.split("@")[0],
        "email": email,
        "password_hash": generate_password_hash(temp_pw),
        "role": role if role in ("admin", "user") else "user",
        "approved": approved,
        "created_at": datetime.utcnow(),
        "last_login": None,
    })

    ok = f"{email} хэрэглэгч нэмэгдлээ (түр нууц үг: {temp_pw})"
    pend = list(users_collection.find({"approved": False}).sort("created_at", DESCENDING))
    actv = list(users_collection.find({"approved": True}).sort("created_at", DESCENDING))
    return render_template("admin_users.html", ok=ok, pending=pend, active=actv)


# ---- Admin: delete user
@app.post("/admin/users/<uid>/delete")
@admin_required
def admin_delete(uid):
    users_collection.delete_one({"_id": ObjectId(uid)})
    return redirect(url_for("admin_users"))


# ---- Admin: promote user -> admin
@app.post("/admin/users/<uid>/promote")
@admin_required
def admin_promote(uid):
    users_collection.update_one({"_id": ObjectId(uid)}, {"$set": {"role": "admin"}})
    return redirect(url_for("admin_users"))


# ---- Admin: demote admin -> user
@app.post("/admin/users/<uid>/demote")
@admin_required
def admin_demote(uid):
    users_collection.update_one({"_id": ObjectId(uid)}, {"$set": {"role": "user"}})
    return redirect(url_for("admin_users"))

@app.post("/admin/users/<uid>/approve")
@admin_required
def admin_approve(uid):
    users_collection.update_one({"_id": ObjectId(uid)}, {"$set": {"approved": True}})
    return redirect(url_for("admin_users"))

@app.post("/admin/users/<uid>/revoke")
@admin_required
def admin_revoke(uid):
    users_collection.update_one({"_id": ObjectId(uid)}, {"$set": {"approved": False}})
    return redirect(url_for("admin_users"))

# ---- Dev seed admin (uses ADMIN_EMAIL + ADMIN_PASSWORD)
@app.post("/_dev/seed-admin")
def seed_admin():
    if os.getenv("ENV", "dev") != "dev":
        abort(404)

    email = (os.getenv("ADMIN_EMAIL") or "").strip().lower()
    if not EMAIL_RE.match(email):
        return {"ok": False, "error": f"ADMIN_EMAIL нь {ALLOW_DOMAIN} домэйноор байх ёстой"}, 400

    admin_pw = os.getenv("ADMIN_PASSWORD")
    if not admin_pw or len(admin_pw) < 6:
        import secrets
        admin_pw = "Adm!" + secrets.token_hex(6)

    u = users_collection.find_one({"email": email})
    if u:
        users_collection.update_one(
            {"_id": u["_id"]},
            {"$set": {"role": "admin", "approved": True, "password_hash": generate_password_hash(admin_pw)}}
        )
        action = "updated"
    else:
        users_collection.insert_one({
            "name": email.split("@")[0],
            "email": email,
            "password_hash": generate_password_hash(admin_pw),
            "role": "admin",
            "approved": True,
            "created_at": datetime.utcnow()
        })
        action = "created"

    log.info("Seed admin %s for %s (approved=True, role=admin)", action, email)
    return {"ok": True, "email": email, "action": action}

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

@app.errorhandler(403)
def forbidden(_):
    return render_template("login.html", error="Энэ хуудас зөвхөн админд нээлттэй."), 403

# ---------------- Scraper downloads ----------------
@app.route("/scraper/download/<fmt>")
@login_required
def scraper_download(fmt: str):
    """
    Scraper-аас үүссэн summary.tsv / summary.xlsx файлуудыг татах линк
    """
    base_dir = Path(__file__).resolve().parent / "scraper" / "_export"
    if fmt == "tsv":
        path = base_dir / "summary.tsv"
    elif fmt == "xlsx":
        path = base_dir / "summary.xlsx"
    else:
        abort(404, "Unknown format")

    if not path.exists():
        abort(404, f"Файл олдсонгүй: {path.name}")

    return send_from_directory(base_dir, path.name, as_attachment=True)

# ---- Main ----
if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=PORT, debug=debug, use_reloader=False, threaded=True)