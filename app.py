# app.py — Unitel AI Hub (Chatbot + Ads Scraper + Sentiment + FB report)
import os, io, sys, subprocess, logging
from pathlib import Path
from threading import Thread
from datetime import datetime, timedelta, date
from urllib.parse import urlparse

import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from pymongo import MongoClient
from dotenv import load_dotenv

# === Chatbot intent / NLU (таны өмнөх processor.py-г ашиглана) ===
import processor

# -------------------- Init --------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = Flask(__name__)
CORS(app)

ROOT_DIR     = Path(__file__).resolve().parent
SCRAPER_DIR  = ROOT_DIR / "scraper"
SUMMARY_TSV  = SCRAPER_DIR / "summary.tsv"
SUMMARY_XLSX = SCRAPER_DIR / "summary.xlsx"
STATIC_DIR   = ROOT_DIR / "static"

STATIC_DIR.mkdir(parents=True, exist_ok=True)

MONGO_URL    = os.getenv("MONGO_URL", "mongodb://localhost:27017/")
FB_TOKEN     = os.getenv("FB_ACCESS_TOKEN", "")
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "")  # /ads/api/* хамгаалалт (optional)

# Mongo
mongo = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
try:
    mongo.admin.command("ping")
    logging.info("✅ Mongo connected")
except Exception as e:
    logging.error("❌ Mongo connection failed: %s", e)
    raise

db       = mongo.unitel_aihub
ads_col  = db.ads_events           # scraper → /ads/api/ingest энд бууна
comments = db.fb_comments          # sentiment demo

# -------------------- Helpers --------------------
def _brand_from_url(u: str) -> str:
    """
    Brand inference: landing/src URL-оос host авч mapping → capitalized fallback.
    """
    BRAND_MAP = {
        "unitel.mn": "Unitel",
        "khanbank.com": "Khan Bank",
        "golomtbank.com": "Golomt Bank",
        "tdbm.mn": "TDB",
        "xacbank.mn": "XacBank",
        "skytel.mn": "Skytel",
        "mobicom.mn": "Mobicom",
        "ardcredit.mn": "ArdCredit",
        "ardholdings.com": "Ard Holdings",
        "coca-cola.mn": "Coca-Cola",
        "pepsi.com": "Pepsi",
    }
    try:
        if not u:
            return ""
        if u.startswith("//"):
            u = "https:" + u
        host = urlparse(u).netloc.lower()
        host = host[4:] if host.startswith("www.") else host
        if host in BRAND_MAP:
            return BRAND_MAP[host]
        label = (host.split(".")[0] or "").replace("-", " ").replace("_", " ").strip()
        return label.capitalize() if label else ""
    except Exception:
        return ""

def _site_home(site: str) -> str:
    if not site:
        return ""
    if site.startswith("http"):
        return site
    return f"https://{site}"

def _iso_date(dt: datetime | None) -> str:
    if not dt:
        return ""
    return dt.strftime("%Y-%m-%d")

def _iso_ts(dt: datetime | None) -> str:
    if not dt:
        return ""
    # real-time HH:mm:ss харуулна (00:00:00 биш)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

# -------------------- Scraper pipeline --------------------
def run_scraper_pipeline_sync() -> dict:
    """
    scraper/run.py → scraper/summarize.py синхрон.
    """
    py = sys.executable
    logging.info("🚀 run.py эхэллээ")
    subprocess.run([py, "run.py"], cwd=str(SCRAPER_DIR), check=True)
    logging.info("🧾 summarize.py эхэллээ")
    subprocess.run([py, "summarize.py"], cwd=str(SCRAPER_DIR), check=True)
    return {"ok": True, "tsv_url": "/report/ads.tsv", "xlsx_url": "/report/ads.xlsx"}

def run_scraper_pipeline_async():
    Thread(target=_bg_scrape, daemon=True).start()

def _bg_scrape():
    try:
        run_scraper_pipeline_sync()
        logging.info("✅ Scraper OK")
    except subprocess.CalledProcessError as e:
        logging.exception("❌ Scraper failed: %s", e)

# -------------------- Chatbot command router --------------------
def chatbot_command_router(message: str) -> dict | None:
    m = (message or "").lower()
    # 1) тайлан шууд өгөх
    if any(k in m for k in ["ads тайлан", "зарын тайлан", "graph тайлан", "facebook тайлан"]):
        msg = (
            "📄 Ads тайлан бэлэн:\n"
            "• TSV: /report/ads.tsv\n"
            "• XLSX: /report/ads.xlsx\n\n"
            "Шинэчлэх бол: “скрэп эхлүүл”"
        )
        return {"response": msg}
    # 2) скрэп эхлүүлэх
    if any(k in m for k in ["скрэп эхлүүл", "scrape", "скрэп", "шинэчлэх"]):
        run_scraper_pipeline_async()
        return {"response": "🧑‍🍳 Скрэп эхлүүллээ. Дуусмагц /admin дээр realtime шинэчлэгдэнэ. TSV/XLSX татаж болно (/report/ads.*)."}
    return None

# -------------------- Pages --------------------
@app.route("/")
def index():
    # Таны index.html-г templates дотор байрлуул (өмнөх загвараа ашигла)
    return render_template("index.html")

@app.route("/admin")
def admin_ads():
    # Доорх admin_ads.html-ийг хэрэглэнэ
    return render_template("admin_ads.html")

# -------------------- Chatbot API --------------------
@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json(silent=True) or request.form
    msg  = (data.get("question") or data.get("message") or "").strip()
    if not msg:
        return jsonify({"response": "🤖 Хоосон асуулт байна."})
    cmd = chatbot_command_router(msg)
    if cmd:
        return jsonify(cmd)
    return jsonify({"response": processor.chatbot_response(msg)})

# -------------------- Admin: realtime summary (Mongo aggregate) --------------------
def _agg_ads(days: int) -> list[dict]:
    """
    ads_events дээрх realtime aggregation – brand, timestamps, counts.
    """
    since = datetime.utcnow() - timedelta(days=days)
    pipeline = [
        {"$match": {"detected_at": {"$gte": since}}},
        {"$sort": {"detected_at": 1}},
        {"$group": {
            "_id": {
                "ad_id": "$ad_id",
                "site": "$site",
                "landing_url": "$url"       # summarize bridge-д url талбар гэж ирдэг
            },
            "first_seen": {"$first": "$detected_at"},
            "last_seen": {"$last": "$detected_at"},
            "times_seen": {"$sum": 1},
            "status": {"$last": "$status"},
            "screenshot": {"$last": "$screenshot"},
            "src_open": {"$last": "$src_open"},
            "landing_open": {"$last": "$url"}  # rename for UI
        }},
        {"$sort": {"last_seen": -1}}
    ]
    items = list(ads_col.aggregate(pipeline))
    out: list[dict] = []
    for it in items:
        ad_id = it["_id"].get("ad_id") or ""
        site  = it["_id"].get("site") or ""
        land  = it["_id"].get("landing_url") or (it.get("landing_open") or "")
        brand = _brand_from_url(land) or _brand_from_url(it.get("src_open", ""))

        first_dt = it.get("first_seen")
        last_dt  = it.get("last_seen")
        out.append({
            "ad_id": ad_id,
            "site": site,
            "site_home": _site_home(site),
            "brand": brand,
            "status": (it.get("status") or "active"),
            "first_seen_date": _iso_date(first_dt),
            "last_seen_date": _iso_date(last_dt),
            "first_seen_ts": _iso_ts(first_dt),
            "last_seen_ts": _iso_ts(last_dt),
            "days_seen": str(((last_dt or datetime.utcnow()).date() - (first_dt or datetime.utcnow()).date()).days + 1 if first_dt and last_dt else 1),
            "times_seen": str(it.get("times_seen") or 1),
            "landing_open": it.get("landing_open") or land,
            "src_open": it.get("src_open") or "",
            "screenshot": it.get("screenshot") or "",
        })
    return out

@app.route("/ads/api/summary")
def api_summary():
    try:
        days = int(request.args.get("days", 30))
    except Exception:
        days = 30
    return jsonify({"ok": True, "items": _agg_ads(days)})

# -------------------- Manual scrape trigger --------------------
@app.route("/admin/scrape-now", methods=["POST"])
def scrape_now():
    run_scraper_pipeline_async()
    return jsonify({"ok": True, "message": "Scraper started"})

# -------------------- Ingest bridge for scraper --------------------
@app.route("/ads/api/upload", methods=["POST"])
def ads_upload():
    # screenshot upload (optional)
    token = request.headers.get("X-INGEST-TOKEN", "")
    if INGEST_TOKEN and token != INGEST_TOKEN:
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    f = request.files.get("file")
    if not f:
        return jsonify({"ok": False, "error": "no file"}), 400

    today = datetime.utcnow().strftime("%Y-%m-%d")
    folder = STATIC_DIR / "ads" / today
    folder.mkdir(parents=True, exist_ok=True)

    fname = f.filename or "ad.png"
    save_path = folder / fname
    f.save(str(save_path))
    rel_url = f"/static/ads/{today}/{fname}"
    return jsonify({"ok": True, "url": rel_url})

@app.route("/ads/api/ingest", methods=["POST"])
def ads_ingest():
    """
    Scraper summarize → Mongo ingest (fast, non-blocking).
    Required: ad_id, site, status
    Optional: url (landing), src_open, screenshot
    """
    token = request.headers.get("X-INGEST-TOKEN", "")
    if INGEST_TOKEN and token != INGEST_TOKEN:
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    for k in ("ad_id", "site", "status"):
        if not data.get(k):
            return jsonify({"ok": False, "error": f"{k} required"}), 400

    # normalize + timestamps
    try:
        dt = datetime.utcnow()
        doc = {
            "ad_id": str(data.get("ad_id")),
            "site": str(data.get("site")),
            "status": str(data.get("status")),
            "url": str(data.get("url") or data.get("landing_url") or ""),
            "src_open": str(data.get("src_open") or ""),
            "screenshot": str(data.get("screenshot") or ""),
            "detected_at": dt,
        }
        ads_col.insert_one(doc)
        return jsonify({"ok": True})
    except Exception as e:
        logging.exception("ingest fail: %s", e)
        return jsonify({"ok": False, "error": "db_error"}), 500

# -------------------- Downloads (TSV/XLSX from scraper) --------------------
@app.route("/report/ads.tsv")
def report_ads_tsv():
    if SUMMARY_TSV.exists():
        return send_file(
            SUMMARY_TSV, as_attachment=True,
            download_name="ads_summary.tsv",
            mimetype="text/tab-separated-values; charset=utf-8"
        )
    # fallback: build from Mongo
    rows = _agg_ads(days=int(request.args.get("days", 30)))
    df = pd.DataFrame(rows)
    out = io.StringIO()
    (df if not df.empty else pd.DataFrame()).to_csv(out, sep="\t", index=False)
    return send_file(
        io.BytesIO(out.getvalue().encode("utf-8-sig")),
        as_attachment=True, download_name="ads_fallback.tsv",
        mimetype="text/tab-separated-values; charset=utf-8"
    )

@app.route("/report/ads.xlsx")
def report_ads_xlsx():
    if SUMMARY_XLSX.exists():
        return send_file(
            SUMMARY_XLSX,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True, download_name="ads_summary.xlsx"
        )
    # fallback: Mongo
    rows = _agg_ads(days=int(request.args.get("days", 30)))
    df = pd.DataFrame(rows)
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        (df if not df.empty else pd.DataFrame()).to_excel(xw, index=False, sheet_name="ads")
    bio.seek(0)
    return send_file(
        bio,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True, download_name="ads_fallback.xlsx"
    )

# -------------------- FB & Sentiment (demo endpoints хэвээр) --------------------
@app.route("/report/fb.<fmt>")
def report_fb(fmt: str):
    days = int(request.args.get("days", 7))
    if not FB_TOKEN:
        empty = pd.DataFrame()
        if fmt == "csv":
            out = io.StringIO(); empty.to_csv(out, index=False)
            return send_file(io.BytesIO(out.getvalue().encode("utf-8-sig")),
                             mimetype="text/csv", as_attachment=True,
                             download_name=f"fb_empty_{days}d.csv")
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as xw:
            empty.to_excel(xw, index=False, sheet_name="fb")
        bio.seek(0)
        return send_file(bio,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True, download_name=f"fb_empty_{days}d.xlsx")
    return jsonify({"ok": True, "note": "FB token байгаа үед бодит өгөгдөл буцаана."})

@app.route("/report/sentiment.<fmt>")
def report_sentiment(fmt: str):
    df = pd.DataFrame(list(comments.find({}, {"_id": 0})))
    if not df.empty and "created_at" in df:
        df["created_at"] = pd.to_datetime(df["created_at"])
    if fmt == "csv":
        out = io.StringIO(); (df if not df.empty else pd.DataFrame()).to_csv(out, index=False)
        return send_file(io.BytesIO(out.getvalue().encode("utf-8-sig")),
                         mimetype="text/csv", as_attachment=True,
                         download_name="sentiment.csv")
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        (df if not df.empty else pd.DataFrame()).to_excel(xw, index=False, sheet_name="sentiment")
    bio.seek(0)
    return send_file(bio,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True, download_name="sentiment.xlsx")

# -------------------- Run --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8888")), debug=True)
