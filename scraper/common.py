# -*- coding: utf-8 -*-
# common.py — TSV storage, hashing, ad-classifier, upsert + helpers

import os, csv, hashlib, requests
from io import BytesIO
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse
from PIL import Image
import imagehash

# =============== Storage (TSV) ===============
DELIM = "\t"
CSV_HEADERS = [
    "site","banner_key","phash","src","landing_url","width","height",
    "first_seen_ts_utc","last_seen_ts_utc","first_seen_date","last_seen_date",
    "days_seen","times_seen","screenshot_path","notes",
    "is_ad","ad_score","ad_reason","ad_id"
]

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def load_db(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path): return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=DELIM))

def save_db(path: str, rows: List[Dict[str, str]]):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=CSV_HEADERS,
            delimiter=DELIM, quoting=csv.QUOTE_MINIMAL, extrasaction="ignore"
        )
        w.writeheader()
        for r in rows:
            # бүх header-ийг заавал бичнэ
            out = {k: (r.get(k, "") if r.get(k) is not None else "") for k in CSV_HEADERS}
            w.writerow(out)
    os.replace(tmp, path)

# =============== HTTP ===============
def http_get_bytes(url: str, timeout=20, referer: Optional[str] = None) -> Optional[bytes]:
    """Зураг татах (GIF-г л үлдээе; WEBP ok)."""
    try:
        if not url: return None
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120",
        }
        if referer: headers["Referer"] = referer
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        if r.status_code != 200: return None
        ct = (r.headers.get("Content-Type") or "").lower()
        if "gif" in ct: return None
        return r.content
    except Exception:
        return None

# =============== Images / Hash ===============
def _img_from_bytes(b: bytes):
    try:
        img = Image.open(BytesIO(b))
        if img.mode not in ("RGB","RGBA","L"): img = img.convert("RGB")
        return img
    except Exception:
        return None

def phash_hex_from_bytes(b: Optional[bytes]) -> str:
    if not b: return ""
    img = _img_from_bytes(b)
    if img is None: return ""
    return str(imagehash.phash(img))

def phash_hex_from_file(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return phash_hex_from_bytes(f.read())
    except Exception:
        return ""

# =============== Time helpers ===============
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def local_today_iso() -> str:
    return datetime.now().astimezone().date().isoformat()

# =============== URL / host ===============
def _host(u: str) -> str:
    try: return urlparse(u).netloc.lower()
    except Exception: return ""

# =============== Ad classifier (heuristics) ===============
_STD_AD_SIZES = [
    (728,90),(970,250),(300,250),(336,280),(300,600),(160,600),
    (320,100),(468,60),(250,250),(240,400),(980,120)
]
_AD_HOST_HINTS = [
    "doubleclick","googlesyndication","googletag","adservice","adserver",
    "smartad","adform","criteo","taboola","outbrain","mgid","teads","banner",
    "/ads/","/banners/","boost.mn","edge.boost.mn"
]
_AD_WORDS = ["ad","ads","advert","sponsor","sponsored","promo","зар","сурталчилгаа"]
_NEG_HINTS = ["thumbnail","/thumb/","/avatars/","/logo","/icons/","/emoji"]

def _near_std_size(w: int, h: int, tol: int = 25) -> bool:
    return any(abs(w-sw)<=tol and abs(h-sh)<=tol for sw,sh in _STD_AD_SIZES)

def classify_ad(site: str, src: str, landing: str, width: str, height: str, notes: str, min_score: int = 3) -> Tuple[str,str,str]:
    """Return (is_ad '1/0', score_str, reason_csv)."""
    score, reasons = 0, []
    w = int(width or 0); h = int(height or 0)
    site_host = _host(f"https://{site}") if site and "://" not in site else _host(site)
    src_host  = _host(src); land_host = _host(landing)
    path = (src or "").lower()

    if "iframe" in (notes or "").lower(): score += 2; reasons.append("iframe")
    if w>=180 and h>=100: score += 1; reasons.append("large")
    if _near_std_size(w,h): score += 2; reasons.append("std_size")
    if any(k in (src_host or "") or k in path for k in _AD_HOST_HINTS): score += 3; reasons.append("ad_host/path")
    if land_host and site_host and land_host != site_host: score += 1; reasons.append("external_click")
    if any(wd in path for wd in _AD_WORDS): score += 1; reasons.append("ad_word")
    if any(neg in path for neg in _NEG_HINTS): score -= 2; reasons.append("editorial_thumb")

    is_ad = "1" if score >= min_score else "0"
    return is_ad, str(score), ",".join(reasons)

# =============== Record / Upsert ===============
def _stable_ad_id(site: str, phash_hex: str, src: str) -> str:
    key = f"{site}|{phash_hex or src}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

class BannerRecord:
    def __init__(self, site, phash_hex, src, landing_url, width, height, screenshot_path, notes, min_ad_score=3):
        self.site, self.phash_hex, self.src, self.landing_url = site, phash_hex, src, landing_url
        self.width, self.height = str(width or 0), str(height or 0)
        self.screenshot_path, self.notes = (screenshot_path or ""), (notes or "")
        self.now_iso, self.today = utc_now_iso(), local_today_iso()
        self.is_ad, self.ad_score, self.ad_reason = classify_ad(site, src, landing_url, self.width, self.height, notes, min_ad_score)
        self.ad_id = _stable_ad_id(site, phash_hex, src)

    @classmethod
    def from_capture(cls, cap: Dict, min_ad_score=3):
        ph = phash_hex_from_bytes(cap.get("img_bytes")) or phash_hex_from_file(cap.get("screenshot_path",""))
        return cls(
            cap.get("site",""), ph, cap.get("src",""), cap.get("landing_url",""),
            cap.get("width"), cap.get("height"), cap.get("screenshot_path",""),
            cap.get("notes",""), min_ad_score
        )

def _nearest_idx(rows: List[Dict[str,str]], site: str, phash_hex: str, thr: int = 5) -> Optional[int]:
    if not phash_hex: return None
    try: new_h = imagehash.hex_to_hash(phash_hex)
    except Exception: return None
    best = None
    for i, r in enumerate(rows):
        if r.get("site") != site: continue
        old_hex = r.get("phash") or ""
        if not old_hex: continue
        try:
            d = imagehash.hex_to_hash(old_hex) - new_h
            if d <= thr:
                best = i if best is None else (i if d < (imagehash.hex_to_hash(rows[best]["phash"]) - new_h) else best)
        except Exception:
            continue
    return best

def upsert_banner(rows: List[Dict[str,str]], rec: BannerRecord) -> Tuple[bool,bool]:
    """Insert or update. Returns (changed, inserted_new)."""
    idx = _nearest_idx(rows, rec.site, rec.phash_hex)
    if idx is None and rec.src:
        for i, r in enumerate(rows):
            if r.get("site")==rec.site and r.get("src")==rec.src:
                idx=i; break

    if idx is None:
        rows.append({
            "site": rec.site, "banner_key": f"{rec.site}:{rec.ad_id}", "phash": rec.phash_hex,
            "src": rec.src, "landing_url": rec.landing_url, "width": rec.width, "height": rec.height,
            "first_seen_ts_utc": rec.now_iso, "last_seen_ts_utc": rec.now_iso,
            "first_seen_date": rec.today, "last_seen_date": rec.today,
            "days_seen": "1", "times_seen": "1", "screenshot_path": rec.screenshot_path,
            "notes": rec.notes, "is_ad": rec.is_ad, "ad_score": rec.ad_score,
            "ad_reason": rec.ad_reason, "ad_id": rec.ad_id,
        })
        return True, True
    else:
        r = rows[idx]
        r["last_seen_ts_utc"] = rec.now_iso
        if r.get("last_seen_date") != rec.today and rec.today:
            r["days_seen"] = str(int(r.get("days_seen","1") or "1")+1)
            r["last_seen_date"] = rec.today
        r["times_seen"] = str(int(r.get("times_seen","0") or "0")+1)
        # хоосон талбаруудыг нөхөх
        if not r.get("landing_url") and rec.landing_url: r["landing_url"] = rec.landing_url
        if not r.get("phash") and rec.phash_hex: r["phash"] = rec.phash_hex
        if not r.get("width") and rec.width: r["width"] = rec.width
        if not r.get("height") and rec.height: r["height"] = rec.height
        if not r.get("screenshot_path") and rec.screenshot_path: r["screenshot_path"] = rec.screenshot_path
        if not r.get("is_ad"): r["is_ad"], r["ad_score"], r["ad_reason"] = rec.is_ad, rec.ad_score, rec.ad_reason
        return True, False
