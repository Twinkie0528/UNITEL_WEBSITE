# summarize.py (Цэвэрлэж, зассан хувилбар)
import csv
import os
from datetime import date
from pathlib import Path

IN_TSV   = "banner_tracking_combined.tsv"
OUT_TSV  = "summary.tsv"
OUT_XLSX = "summary.xlsx"
DELIM    = "\t"

ALLOW_SITES = {"gogo.mn", "ikon.mn", "news.mn"}
END_ABSENCE_DAYS = 2

def file_url(p):
    if not p: return ""
    return "file:///" + str(Path(p).resolve()).replace("\\", "/")

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
        ls = date.fromisoformat(last_seen_date_str)
        return "ИДЭВХТЭЙ" if (date.today() - ls).days < END_ABSENCE_DAYS else "ДУУССАН"
    except Exception:
        return "ИДЭВХТЭЙ"

if not os.path.exists(IN_TSV):
    print(f"Оролтын файл '{IN_TSV}' олдсонгүй — эхлээд python run.py ажиллуулна уу.")
    sys.exit(0)

with open(IN_TSV, mode='r', newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter=DELIM))

ads = [r for r in rows if r.get("is_ad") == "1" and r.get("site") in ALLOW_SITES]

if not ads:
    print(f"'{IN_TSV}' дотор is_ad=1 гэсэн мөр олдсонгүй.")
    # Хоосон ч гэсэн тайлангийн файлуудыг үүсгэе
    with open(OUT_TSV, "w", encoding="utf-8") as f: f.write("")
    with open(OUT_XLSX, "w", encoding="utf-8") as f: f.write("")
    sys.exit(0)

out_rows = []
for r in ads:
    site = r.get("site", "")
    src = r.get("src", "")
    land = r.get("landing_url", "")
    ls = r.get("last_seen_date", "")
    status = _status_by_last_seen(ls)
    src_open = normalize_url(src, site)
    land_open = normalize_url(land, site)
    src_display = src_open or src
    landing_display = land_open or land

    out_rows.append({
        "site": site,
        "status": status,
        "first_seen_date": r.get("first_seen_date", ""),
        "last_seen_date":  r.get("last_seen_date", ""),
        "days_seen":       r.get("days_seen", "0"),
        "times_seen":      r.get("times_seen", "0"),
        "ad_score":        r.get("ad_score", ""),
        "ad_reason":       r.get("ad_reason", ""),
        "src":             src_display,
        "src_open":        src_open,
        "landing_url":     landing_display,
        "landing_open":    land_open,
        "screenshot_path": r.get("screenshot_path", ""),
        "screenshot_file": file_url(r.get("screenshot_path", "")),
    })

# ===== TSV бичих =====
with open(OUT_TSV, "w", newline="", encoding="utf-8") as f:
    cols = list(out_rows[0].keys())
    w = csv.DictWriter(f, fieldnames=cols, delimiter=DELIM)
    w.writeheader()
    w.writerows(out_rows)
print(f"Амжилттай бичлээ: {OUT_TSV} ({len(out_rows)} мөр)")

# ===== XLSX бичих =====
try:
    import xlsxwriter
    with xlsxwriter.Workbook(OUT_XLSX) as wb:
        ws = wb.add_worksheet("Ads")
        bold = wb.add_format({"bold": True})
        urlfm = wb.add_format({"font_color": "blue", "underline": 1})

        cols = list(out_rows[0].keys())
        ws.write_row(0, 0, cols, bold)

        for r_i, r in enumerate(out_rows, start=1):
            for c_i, h in enumerate(cols):
                val = r[h]
                if h == "src_open" and val:
                    ws.write_url(r_i, c_i, val, urlfm, r["src"] or val)
                elif h == "landing_open" and val:
                    ws.write_url(r_i, c_i, val, urlfm, r["landing_url"] or val)
                elif h == "screenshot_file" and val:
                    ws.write_url(r_i, c_i, val, urlfm, "open")
                else:
                    ws.write(r_i, c_i, val)
    print(f"Амжилттай бичлээ: {OUT_XLSX}")
except ModuleNotFoundError:
    print("Анхааруулга: xlsxwriter суугаагүй тул XLSX файл үүсгэсэнгүй. (pip install xlsxwriter)")