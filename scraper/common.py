# -*- coding: utf-8 -*-
# common.py (v4) — TSV storage + robust HTTP + EXIF/animated handling + brand extraction + pHash config + ad-classifier + upsert

import os
import csv
import hashlib
import requests
import re
from io import BytesIO
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse, unquote
from PIL import Image, ImageOps
import imagehash

# ===================== Config =====================
DELIM = "\t"
PHASH_SIZE = int(os.getenv("PHASH_SIZE", "8"))             # 8 эсвэл 16
HAMMING_THR = int(os.getenv("PHASH_HAMMING_THR", "8"))     # 8..12 орчим
LOCAL_TZ = os.getenv("LOCAL_TZ", "Asia/Ulaanbaatar")
MAX_BYTES_DEFAULT = int(os.getenv("HTTP_MAX_BYTES", "5000000"))  # 5MB

# ===================== Storage (TSV) =====================
CSV_HEADERS = [
    "site", "brand", "banner_key", "phash", "src", "landing_url", "width", "height",
    "first_seen_ts_utc", "last_seen_ts_utc", "first_seen_date", "last_seen_date",
    "days_seen", "times_seen", "screenshot_path", "notes",
    "is_ad", "ad_score", "ad_reason", "ad_id"
]

def ensure_dir(p: str):
    """Замын хавтсыг үүсгэнэ; файлын зам ирсэн бол түүний директорыг үүсгэнэ."""
    if not p: return
    base, ext = os.path.splitext(p)
    d = os.path.dirname(p) if ext else p
    if d: os.makedirs(d, exist_ok=True)

def load_db(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path): return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=DELIM))

def save_db(path: str, rows: List[Dict[str, str]]):
    """Аюулгүй бичих; portalocker байх бол богино лок хэрэглэнэ."""
    ensure_dir(path)
    tmp = path + ".tmp"

    # Optional file lock
    lock_path = path + ".lock"
    try:
        import portalocker  # optional
        with portalocker.Lock(lock_path, timeout=5):
            _write_rows(tmp, rows)
            os.replace(tmp, path)
    except Exception:
        # lock байхгүй/алдаатай бол энгийн бичих горим
        _write_rows(tmp, rows)
        os.replace(tmp, path)

def _write_rows(tmp: str, rows: List[Dict[str, str]]):
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=CSV_HEADERS,
            delimiter=DELIM, quoting=csv.QUOTE_MINIMAL, extrasaction="ignore"
        )
        w.writeheader()
        for r in rows:
            out = {k: (r.get(k, "") if r.get(k) is not None else "") for k in CSV_HEADERS}
            w.writerow(out)

# ===================== HTTP (Session + Retry + Stream cap) =====================
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session = requests.Session()
_session.headers.update({
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
})
_retry = Retry(total=3, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504])
_adapter = HTTPAdapter(max_retries=_retry, pool_connections=20, pool_maxsize=20)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)

_SKIP_MIME = ("gif", "svg", "xml")
_SKIP_EXT  = (".gif", ".svg", ".xml")

def http_get_bytes(url: str, timeout: int = 15, referer: Optional[str] = None,
                   max_bytes: int = MAX_BYTES_DEFAULT) -> Optional[bytes]:
    """
    Зураг/бичлэг татах — stream-ээр уншиж бодитоор max_bytes хязгаарлана.
    SSRF/локал хаягуудыг хаана. GIF/SVG/XML алгасна.
    """
    if not url or url.startswith(("data:", "file:", "ftp:")):
        return None
    low = url.lower()
    if any(low.endswith(ext) for ext in _SKIP_EXT):
        return None
    # SSRF/локал хамгаалалт
    try:
        host = urlparse(url).hostname or ""
        if host in {"localhost"} or host.startswith(("127.", "169.254.")):
            return None
    except Exception:
        return None

    headers = {}
    if referer:
        headers["Referer"] = referer

    try:
        with _session.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True) as r:
            if r.status_code != 200:
                return None
            ct = (r.headers.get("Content-Type") or "").lower()
            if any(m in ct for m in _SKIP_MIME):
                return None
            cl = r.headers.get("Content-Length")
            if cl and max_bytes and int(cl) > max_bytes:
                return None

            buf = BytesIO()
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                buf.write(chunk)
                if max_bytes and buf.tell() > max_bytes:
                    return None
            return buf.getvalue()
    except Exception:
        return None

# ===================== Images / Hash (+EXIF transpose, animated) =====================
def _img_from_bytes(b: bytes):
    try:
        img = Image.open(BytesIO(b))
        # EXIF orientation засах
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        return img
    except Exception:
        return None

def is_animated_image_bytes(b: Optional[bytes]) -> bool:
    if not b: return False
    img = _img_from_bytes(b)
    if not img: return False
    try:
        return getattr(img, "is_animated", False) or getattr(img, "n_frames", 1) > 1
    except Exception:
        return False

def phash_hex_from_bytes(b: Optional[bytes]) -> str:
    if not b: return ""
    img = _img_from_bytes(b)
    if img is None: return ""
    return str(imagehash.phash(img, hash_size=PHASH_SIZE))

def phash_hex_from_file(path: str) -> str:
    if not path or not os.path.exists(path): return ""
    try:
        with open(path, "rb") as f:
            return phash_hex_from_bytes(f.read())
    except Exception:
        return ""

# ===================== Time helpers =====================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def local_today_iso() -> str:
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(LOCAL_TZ)
        return datetime.now(tz).date().isoformat()
    except Exception:
        return datetime.now().astimezone().date().isoformat()

# ===================== URL / host / brand =====================
def _host(u: str) -> str:
    try: return urlparse(u).netloc.lower()
    except Exception: return ""

def _norm_site_host(site: str) -> str:
    """'gogo.mn' эсвэл 'https://gogo.mn' → 'gogo.mn'."""
    if not site: return ""
    if "://" not in site: site = "https://" + site
    return _host(site)

# Нийтлэг TLD/Subdomain ба ad network-ууд
_COMMON_TLDS = {".com", ".mn", ".org", ".net", ".info", ".biz", ".gov", ".edu", ".co", ".io"}
_COMMON_SUBDOMAINS = {"www", "m", "mobile", "shop", "promo", "news", "blog", "en"}
_AD_NETWORKS = {
    "google": {"doubleclick.net", "googlesyndication.com", "googleadservices.com"},
    "facebook": {"facebook.com", "fb.com"},
    "criteo": {"criteo.com"},
    "taboola": {"taboola.com"},
}

def _extract_true_url_from_redirect(u: str) -> Optional[str]:
    """Google/Facebook redirect query-оос бодит landing URL-ыг ил гаргах оролдлого."""
    try:
        q = urlparse(u).query
        pairs = re.findall(r'(?:^|&)([^=]+)=([^&]+)', q)
        params = {k: v for k, v in pairs}
    except Exception:
        params = {}
    for key in ("adurl", "url", "u", "redirect", "rd", "to"):
        if key in params:
            cand = unquote(params[key])
            if cand.startswith("http"):
                return cand
    return None

def extract_brand_from_url(url: str) -> str:
    """
    Landing URL-оос брэндийн нэрийг таамаглана.
    - Redirect query (adurl/url/u/...) байвал бодит URL руу “өөдрүүлнэ”
    - Ad network домэйноос “жинхэнэ линк”-ийг дахин шалгана
    - tldextract байвал ашиглана (optional), үгүй бол heuristic
    """
    if not url: return ""
    try:
        real = _extract_true_url_from_redirect(url) or url
        hostname = urlparse(real).hostname
        if not hostname: return ""
        hostname = hostname.lower()

        # Ad network илэрвэл дахин бодит линк хайна
        for network, domains in _AD_NETWORKS.items():
            if any(d in hostname for d in domains):
                cand = _extract_true_url_from_redirect(real)
                if cand:
                    hostname = (urlparse(cand).hostname or hostname).lower()
                else:
                    return network.title()

        # Илүү найдвартай бол tldextract ашиглана
        brand = ""
        try:
            import tldextract  # optional
            ext = tldextract.extract(hostname)
            # ext.domain ихэвчлэн брэндэд хамгийн ойр
            brand = ext.domain or ""
        except Exception:
            # Heuristic: дэд домэйнүүдийг шүүж meaningful хэсгээс сүүлчийнхийг авна
            parts = hostname.split(".")
            meaningful_parts = [p for p in parts if ("."+p) not in _COMMON_TLDS and p not in _COMMON_SUBDOMAINS]
            if meaningful_parts:
                cands = [p for p in meaningful_parts if len(p) > 1 and not p.isdigit()]
                brand = cands[-1] if cands else meaningful_parts[-1]
            else:
                brand = parts[-2] if len(parts) >= 2 else hostname

        return brand
    except Exception:
        return ""

# ===================== Ad classifier =====================
_STD_AD_SIZES = [(728,90),(970,250),(300,250),(336,280),(300,600),(160,600),(320,100),(468,60),(250,250),(240,400),(980,120)]
_AD_HOST_HINTS = ["doubleclick","googlesyndication","googletag","adservice","adserver","smartad","adform","criteo","taboola","outbrain","mgid","teads","banner","/ads/","/banners/","boost.mn","edge.boost.mn"]
_AD_WORDS = ["ad","ads","advert","sponsor","sponsored","promo","зар","сурталчилгаа"]
_NEG_HINTS = ["thumbnail","/thumb/","/avatars/","/logo","/icons/","/emoji","/share/","/social/"]

def _near_std_size(w: int, h: int, tol: int = 25) -> bool:
    return any(abs(w-sw)<=tol and abs(h-sh)<=tol for sw,sh in _STD_AD_SIZES)

def classify_ad(site: str, src: str, landing: str, width: str, height: str, notes: str,
                min_score: int = 5, img_bytes: Optional[bytes] = None) -> Tuple[str,str,str]:
    """
    Return (is_ad '1/0', score_str, reason_csv).
    Оноо (heuristics):
      - iframe/banner notes
      - том хэмжээ + стандарт хэмжээ
      - ad host/path keywords
      - өөр домэйн рүү үсрэлт
      - ad-related үгс
      - thumbnail/logo/icon зэрэг сөрөг оноо
      - aspect ratio (өргөн тууз, босоо баннер)
      - animated hint (gif/mp4/webm path эсвэл animated image)
    Guardrail: min_score≤4 үед (ad-host ИЛЭРХИЙ эсвэл external эсвэл стандарт хэмжээ) байхгүй бол ad биш болгоно.
    """
    score, reasons = 0, []
    w = int(width or 0); h = int(height or 0)
    site_host = _norm_site_host(site)
    src_host  = _host(src); land_host = _host(landing)
    path = (src or "").lower()
    note_l = (notes or "").lower()

    # Notes-based
    if "iframe" in note_l: score += 2; reasons.append("iframe")
    if any(k in note_l for k in ["banner","ad","promo"]): score += 1; reasons.append("banner_note")

    # Size
    if w >= 200 and h >= 100: score += 1; reasons.append("large")
    if _near_std_size(w,h): score += 2; reasons.append("std_size")

    # Host/path hints
    if any(k in (src_host or "") or k in path for k in _AD_HOST_HINTS):
        score += 3; reasons.append("ad_host/path")

    # External landing
    if land_host and site_host and land_host != site_host:
        score += 2; reasons.append("external_click")

    # Words in filename
    if any(wd in path for wd in _AD_WORDS):
        score += 1; reasons.append("ad_word")

    # Negative hints
    if any(neg in path for neg in _NEG_HINTS):
        score -= 3; reasons.append("editorial_thumb")

    # Aspect ratio (banner-like)
    if w > 0 and h > 0:
        ratio = (w / h) if h != 0 else 0
        if 4 < ratio < 6:          # 728x90, 980x120 гэх өргөн тууз
            score += 1; reasons.append("wide_ratio")
        if 0.2 < ratio < 0.35:     # 300x600 гэх босоо баннер
            score += 1; reasons.append("tall_ratio")

    # Animated hint (file path + бодитоор animated эсэх)
    if any(ext in path for ext in [".gif", ".mp4", ".webm"]):
        score += 1; reasons.append("animated_hint")
    elif img_bytes and is_animated_image_bytes(img_bytes):
        score += 1; reasons.append("animated_hint")

    # Эхний шийд
    is_ad = "1" if score >= min_score else "0"

    # Guardrail: min_score≤4 үед thumbnail/logo-оос хамгаалах
    if is_ad == "1" and min_score <= 4 and score <= 4:
        has_ad_host = any(k in (src_host or "") or k in path for k in _AD_HOST_HINTS)
        external    = (land_host and site_host and land_host != site_host)
        near_std    = _near_std_size(w, h)
        if not (has_ad_host or external or near_std):
            is_ad = "0"
            reasons.append("guard_low_score")

    return is_ad, str(score), ",".join(reasons)

# ===================== Record / Upsert =====================
def _stable_ad_id(site: str, phash_hex: str, src: str) -> str:
    key = f"{site}|{phash_hex or src}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

class BannerRecord:
    def __init__(self, site, phash_hex, src, landing_url, width, height,
                 screenshot_path, notes, min_ad_score=5, img_bytes: Optional[bytes]=None):
        self.site, self.phash_hex, self.src, self.landing_url = site, phash_hex, src, landing_url
        self.width, self.height = str(width or 0), str(height or 0)
        self.screenshot_path, self.notes = (screenshot_path or ""), (notes or "")
        self.brand = extract_brand_from_url(landing_url)
        self.now_iso, self.today = utc_now_iso(), local_today_iso()
        self.is_ad, self.ad_score, self.ad_reason = classify_ad(
            site, src, landing_url, self.width, self.height, notes, min_ad_score, img_bytes=img_bytes
        )
        self.ad_id = _stable_ad_id(site, phash_hex, src)

    @classmethod
    def from_capture(cls, cap: Dict, min_ad_score=5):
        """
        cap: {
          site, src, landing_url, width, height, screenshot_path, notes, img_bytes?
        }
        """
        img_b = cap.get("img_bytes")
        ph = phash_hex_from_bytes(img_b) or phash_hex_from_file(cap.get("screenshot_path",""))
        return cls(
            cap.get("site",""), ph, cap.get("src",""), cap.get("landing_url",""),
            cap.get("width"), cap.get("height"), cap.get("screenshot_path",""),
            cap.get("notes",""), min_ad_score, img_bytes=img_b
        )

def _nearest_idx(rows: List[Dict[str,str]], site: str, phash_hex: str, thr: int = None) -> Optional[int]:
    """
    pHash зай бага (<= thr) байвал ижил баннер гэж үзэж update хийнэ.
    thr default = HAMMING_THR (env-оос).
    """
    if not phash_hex: return None
    thr = thr if thr is not None else HAMMING_THR
    try:
        new_h = imagehash.hex_to_hash(phash_hex)
    except Exception:
        return None

    best, best_d = None, None
    for i, r in enumerate(rows):
        if r.get("site") != site: continue
        old_hex = r.get("phash") or ""
        if not old_hex: continue
        try:
            d = imagehash.hex_to_hash(old_hex) - new_h
        except Exception:
            continue
        if d <= thr and (best is None or d < best_d):
            best, best_d = i, d
    return best

def upsert_banner(rows: List[Dict[str,str]], rec: BannerRecord) -> Tuple[bool,bool]:
    """Insert or update. Returns (changed, inserted_new)."""
    idx = _nearest_idx(rows, rec.site, rec.phash_hex)

    # pHash байхгүй бол SRC-ээр fallback
    if idx is None and rec.src:
        for i, r in enumerate(rows):
            if r.get("site")==rec.site and r.get("src")==rec.src:
                idx = i
                break

    if idx is None:
        # Insert
        rows.append({
            "site": rec.site, "brand": rec.brand, "banner_key": f"{rec.site}:{rec.ad_id}", "phash": rec.phash_hex,
            "src": rec.src, "landing_url": rec.landing_url, "width": rec.width, "height": rec.height,
            "first_seen_ts_utc": rec.now_iso, "last_seen_ts_utc": rec.now_iso,
            "first_seen_date": rec.today, "last_seen_date": rec.today,
            "days_seen": "1", "times_seen": "1", "screenshot_path": rec.screenshot_path,
            "notes": rec.notes, "is_ad": rec.is_ad, "ad_score": rec.ad_score,
            "ad_reason": rec.ad_reason, "ad_id": rec.ad_id,
        })
        return True, True
    else:
        # Update
        r = rows[idx]; changed = False
        old_last = r.get("last_seen_date","")
        r["last_seen_ts_utc"] = rec.now_iso
        if old_last != rec.today and rec.today:
            r["days_seen"] = str(int(r.get("days_seen","1") or "1") + 1); changed = True
            r["last_seen_date"] = rec.today
        r["times_seen"] = str(int(r.get("times_seen","0") or "0") + 1); changed = True

        if not r.get("landing_url") and rec.landing_url:
            r["landing_url"] = rec.landing_url; changed = True
        if not r.get("phash") and rec.phash_hex:
            r["phash"] = rec.phash_hex; changed = True
        if (not r.get("width") or r.get("width") == "0") and rec.width:
            r["width"] = rec.width; changed = True
        if (not r.get("height") or r.get("height") == "0") and rec.height:
            r["height"] = rec.height; changed = True
        if not r.get("screenshot_path") and rec.screenshot_path:
            r["screenshot_path"] = rec.screenshot_path; changed = True
        if not r.get("brand") and rec.brand:
            r["brand"] = rec.brand; changed = True

        try:
            old_score = int(r.get("ad_score","0") or "0")
            new_score = int(rec.ad_score or "0")
        except Exception:
            old_score, new_score = 0, 0
        if new_score > old_score:
            r["is_ad"], r["ad_score"], r["ad_reason"] = rec.is_ad, rec.ad_score, rec.ad_reason; changed = True
        if not r.get("is_ad"):
            r["is_ad"], r["ad_score"], r["ad_reason"] = rec.is_ad, rec.ad_score, rec.ad_reason; changed = True

        return changed, False
