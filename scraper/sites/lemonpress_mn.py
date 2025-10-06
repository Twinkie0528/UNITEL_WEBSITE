# scraper/sites/lemonpress_mn.py
from pathlib import Path
from typing import List, Dict, Any
import time, hashlib, requests
from bs4 import BeautifulSoup
from .utils import norm, pick_src_from_img, first_external_link
from common import http_get_bytes

HOME = "https://lemonpress.mn"
CAT  = "https://lemonpress.mn/category/surtalchilgaa"

def scrape_lemonpress(output_dir: str | Path, *, dwell_seconds: int = 0, headless: bool = True) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    out_dir = Path(output_dir) / "lemonpress.mn"
    out_dir.mkdir(parents=True, exist_ok=True)

    r = requests.get(CAT, headers={"User-Agent":"Mozilla/5.0","Accept-Language":"mn,en;q=0.8"}, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    # Картууд – article/a дотор зурагтай линк
    cards = soup.select("article a[href*='/post/'], a.card:has(img), article a:has(img)")
    seen = set()

    for i, a in enumerate(cards):
        href = norm(a.get("href",""), HOME)
        img = a.find("img")
        src = pick_src_from_img(img, HOME)
        if not src:
            s = a.select_one("source[srcset]")
            if s and s.get("srcset"):
                src = norm(s["srcset"].split(",")[0].strip().split(" ")[0], HOME)
        if not src or src in seen:
            continue
        seen.add(src)

        # Пост руу орж гадны landing олох
        landing = ""
        try:
            ra = requests.get(href, headers={"User-Agent":"Mozilla/5.0","Referer":CAT}, timeout=20)
            if ra.ok:
                soup_a = BeautifulSoup(ra.text, "lxml")
                landing = first_external_link(soup_a, "lemonpress.mn") or href
        except Exception:
            landing = href

        img_bytes = http_get_bytes(src, referer=HOME)
        fname = f"lemonpress_{int(time.time())}_{i}_{hashlib.md5(src.encode()).hexdigest()[:8]}.png"
        shot  = str(out_dir / fname)
        if img_bytes:
            Path(shot).write_bytes(img_bytes)

        out.append({
            "site":"lemonpress.mn",
            "src":src,
            "landing_url":landing,
            "landing_final_url":landing,
            "img_bytes":img_bytes,
            "width":0,"height":0,
            "screenshot_path":shot,
            "context":"category=surtalchilgaa",
        })
    return out
