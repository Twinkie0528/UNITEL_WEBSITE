# -*- coding: utf-8 -*-
"""
summarize.py (v4 - Файлын замыг зөв тодорхойлсон, сайжруулсан хувилбар)
- banner_tracking_combined.tsv -> scraper/_export/summary.tsv + scraper/_export/summary.xlsx
- .env болон командын мөрөөс тохиргоо уншина
- XLSX бичих үед PermissionError (файл нээлттэй) бол timestamp fallback
- URL normalize + file:/// local screenshot линк
"""

import os
import sys
import csv
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Set

# --- .env (optional) ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# <<< ШИНЭЧЛЭЛТ 1: Файлын замыг зөв, ухаалаг тодорхойлох хэсэг >>>
# Энэ скрипт байгаа газрыг (scraper/ хавтас) тодорхойлох
SCRIPT_DIR = Path(__file__).resolve().parent
# Гаралтын файлууд хадгалагдах _export хавтасны замыг тодорхойлох
EXPORT_DIR = SCRIPT_DIR / "_export"
# Хэрэв _export хавтас байхгүй бол урьдчилан үүсгэх
EXPORT_DIR.mkdir(exist_ok=True)

# ------------------------------
# Defaults / ENV overrides
# ------------------------------
# <<< ШИНЭЧЛЭЛТ 2: Гаралтын файлын нэрэнд шинээр тодорхойлсон замыг ашиглах >>>
DEF_IN_TSV   = os.getenv("IN_TSV",   SCRIPT_DIR / "banner_tracking_combined.tsv")
DEF_OUT_TSV  = os.getenv("OUT_TSV",  EXPORT_DIR / "summary.tsv")
DEF_OUT_XLSX = os.getenv("OUT_XLSX", EXPORT_DIR / "summary.xlsx")
DEF_DELIM    = "\t"
DEF_ALLOW = os.getenv(
    "ALLOW_SITES",
    "gogo.mn,ikon.mn,news.mn,caak.mn,ublife.mn,lemonpress.mn,bolor-toli.com"
)
DEF_ABSENCE  = int(os.getenv("END_ABSENCE_DAYS", "2"))
DEF_MIN_SCORE = int(os.getenv("SUMMARY_MIN_SCORE", "0"))

# ------------------------------
# CLI (Command Line Interface)
# ------------------------------
p = argparse.ArgumentParser(description="Summarize ad ledger (TSV -> TSV+XLSX).")
p.add_argument("--in-tsv",  default=DEF_IN_TSV,   help="Оролтын TSV (ledger)")
p.add_argument("--out-tsv", default=DEF_OUT_TSV,  help="Гаралтын TSV (summary)")
p.add_argument("--out-xlsx",default=DEF_OUT_XLSX, help="Гаралтын XLSX (summary)")
p.add_argument("--allow-sites", default=DEF_ALLOW, help="Шүүх сайтууд (comma-separated). --all-sites тавибал игнорлоно.")
p.add_argument("--all-sites", action="store_true", help="Сайт шүүлтгүй (ALLOW_SITES-ийг үл тооцно).")
p.add_argument("--absence-days", type=int, default=DEF_ABSENCE, help="Идэвхгүй гэж тооцох хоног.")
p.add_argument("--min-score", type=int, default=DEF_MIN_SCORE, help="Давхар шүүлт: ad_score >= min-score (is_ad=1 дээр нэмэлт).")
args = p.parse_args()

IN_TSV   = Path(args.in_tsv)
OUT_TSV  = Path(args.out_tsv)
OUT_XLSX = Path(args.out_xlsx)
DELIM    = DEF_DELIM
END_ABSENCE_DAYS = args.absence_days

ALLOW_SITES: Set[str] = set()
if not args.all_sites:
    ALLOW_SITES = {s.strip() for s in (args.allow_sites or "").split(",") if s.strip()}

# ------------------------------
# Helpers
# ------------------------------
def file_url(p: str) -> str:
    if not p: return ""
    # Pathlib ашиглан OS-ээс хамааралгүй зөв URL үүсгэх
    return Path(p).resolve().as_uri()

def normalize_url(u: str, site: str) -> str:
    if not u: return ""
    s = u.strip()
    if s.startswith(("http://", "https://", "file:///")): return s
    if s.startswith("//"): return "https:" + s
    if s.startswith("/"):
        base = site if site.startswith("http") else ("https://" + site)
        return base.rstrip("/") + s
    if "." in s and " " not in s and not s.startswith(("iframe://", "bg://", "video://")):
        return "https://" + s
    return ""

def _status_by_last_seen(last_seen_date_str: str) -> str:
    try:
        ls = datetime.date.fromisoformat(last_seen_date_str)
        return "ИДЭВХТЭЙ" if (datetime.date.today() - ls).days < END_ABSENCE_DAYS else "ДУУССАН"
    except Exception:
        return "ИДЭВХТЭЙ"

def read_ledger(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        print(f"Оролтын файл '{path}' олдсонгүй — эхлээд `python run.py` ажиллуулна уу.")
        sys.exit(0)
    with path.open(mode="r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=DELIM))

def write_tsv(path: Path, rows: List[Dict[str, str]]):
    with path.open("w", newline="", encoding="utf-8") as f:
        cols = list(rows[0].keys()) if rows else []
        w = csv.DictWriter(f, fieldnames=cols, delimiter=DELIM)
        w.writeheader()
        w.writerows(r for r in rows)

def safe_write_xlsx(path: Path, rows: List[Dict[str, str]]):
    try:
        import xlsxwriter
    except ModuleNotFoundError:
        print("Анхааруулга: xlsxwriter суугаагүй тул XLSX файл үүсгэсэнгүй. (pip install xlsxwriter)")
        return

    def _write(wb):
        ws = wb.add_worksheet("Ads")
        if not rows: return
        bold = wb.add_format({"bold": True})
        urlfm = wb.add_format({"font_color": "blue", "underline": 1})
        cols = list(rows[0].keys())
        ws.write_row(0, 0, cols, bold)
        for r_i, r in enumerate(rows, start=1):
            for c_i, h in enumerate(cols):
                val = r.get(h, "")
                if h in ("src_open", "landing_open", "screenshot_file") and val:
                    link_text = "open" if h == "screenshot_file" else (r.get(h[:-5], val))
                    ws.write_url(r_i, c_i, val, urlfm, link_text)
                else:
                    ws.write(r_i, c_i, val)
        for c_i, h in enumerate(cols):
            maxlen = max([len(str(h))] + [len(str(r.get(h, ""))) for r in rows])
            ws.set_column(c_i, c_i, min(60, max(12, maxlen * 0.9)))

    try:
        with xlsxwriter.Workbook(path) as wb:
            _write(wb)
        print(f"✅ Амжилттай бичлээ: {path}")
    except Exception as e:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_path = path.with_name(f"{path.stem}_{ts}{path.suffix}")
        try:
            with xlsxwriter.Workbook(alt_path) as wb:
                _write(wb)
            print(f"⚠️ Анхааруулга: '{path}' түгжигдсэн байж магадгүй. Файлыг {alt_path} нэрээр үүсгэлээ.")
        except Exception as e2:
            print(f"❌ XLSX файл бичихэд алдаа гарлаа: {e2}")

# ------------------------------
# Run
# ------------------------------
rows = read_ledger(IN_TSV)

ads = []
for r in rows:
    if r.get("is_ad") != "1":
        continue
    if not args.all_sites and ALLOW_SITES and r.get("site") not in ALLOW_SITES:
        continue
    if args.min_score and int(r.get("ad_score") or 0) < args.min_score:
        continue
    ads.append(r)

if not ads:
    print(f"'{IN_TSV}' дотор тохирох мөр олдсонгүй (шүүлтүүрт тохироогүй).")
    OUT_TSV.write_text("", encoding="utf-8")
    safe_write_xlsx(OUT_XLSX, [])
    sys.exit(0)

out_rows: List[Dict[str, str]] = []
OUT_HEADERS = [
    "site", "brand", "status", "first_seen_date", "last_seen_date",
    "days_seen", "times_seen", "ad_score", "ad_reason", "src",
    "src_open", "landing_url", "landing_open", "screenshot_path",
    "screenshot_file"
]

for r in ads:
    site = r.get("site", "")
    src = r.get("src", "")
    land = r.get("landing_url", "")
    ls = r.get("last_seen_date", "") or r.get("first_seen_date", "")
    
    out_rows.append({
        "site": site,
        "brand": r.get("brand", ""),
        "status": _status_by_last_seen(ls),
        "first_seen_date": r.get("first_seen_date", ""),
        "last_seen_date":  r.get("last_seen_date", ""),
        "days_seen":       r.get("days_seen", "0"),
        "times_seen":      r.get("times_seen", "0"),
        "ad_score":        r.get("ad_score", ""),
        "ad_reason":       r.get("ad_reason", ""),
        "src":             normalize_url(src, site) or src,
        "src_open":        normalize_url(src, site),
        "landing_url":     normalize_url(land, site) or land,
        "landing_open":    normalize_url(land, site),
        "screenshot_path": r.get("screenshot_path", ""),
        "screenshot_file": file_url(r.get("screenshot_path", "")),
    })

# ===== TSV бичих =====
# `write_tsv` функцэд header-г зөв дамжуулахын тулд багана үүсгэх
final_rows_for_tsv = []
for row in out_rows:
    new_row = {h: row.get(h, "") for h in OUT_HEADERS}
    final_rows_for_tsv.append(new_row)

write_tsv(OUT_TSV, final_rows_for_tsv)
print(f"✅ Амжилттай бичлээ: {OUT_TSV} ({len(out_rows)} мөр)")

# ===== XLSX бичих =====
safe_write_xlsx(OUT_XLSX, out_rows)