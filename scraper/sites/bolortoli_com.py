# scraper/sites/bolortoli_com.py
from pathlib import Path
from typing import List, Dict, Any
import time, hashlib, requests
from bs4 import BeautifulSoup
from .utils import norm, pick_src_from_img, pick_bg_from_style
from common import http_get_bytes

HOME = "https://bolor-toli.com"

def scrape_bolortoli(output_dir: str | Path, *, dwell_seconds: int = 0, headless: bool = True) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    out_dir = Path(output_dir) / "bolor-toli.com"
    out_dir.mkdir(parents=True, exist_ok=True)

    r = requests.get(HOME, headers={"User-Agent":"Mozilla/5.0","Accept-Language":"mn,en;q=0.8"}, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    # Слайдер & промо хэсгүүд
    nodes = []
    nodes += soup.select("div[id*='slider'] a, div[class*='slider'] a, div[id*='carousel'] a, div[class*='carousel'] a")
    nodes += soup.select("a[href] img")

    seen = set()
    for i, node in enumerate(nodes):
        a = node if node.name == "a" else node.find_parent("a")
        if not a: continue

        href = norm(a.get("href",""), HOME)

        src = ""
        img = a.find("img")
        if img: src = pick_src_from_img(img, HOME)
        if not src:
            # background-image дээр
            src = pick_bg_from_style(a.get("style",""), HOME)
            if not src:
                inner = a.select_one("[style*='background']")
                if inner:
                    src = pick_bg_from_style(inner.get("style",""), HOME)
        if not src or src in seen:
            continue
        seen.add(src)

        img_bytes = http_get_bytes(src, referer=HOME)
        fname = f"bolortoli_{int(time.time())}_{i}_{hashlib.md5(src.encode()).hexdigest()[:8]}.png"
        shot  = str(out_dir / fname)
        if img_bytes:
            Path(shot).write_bytes(img_bytes)

        out.append({
            "site":"bolor-toli.com",
            "src":src,
            "landing_url":href or HOME,
            "landing_final_url":href or HOME,
            "img_bytes":img_bytes,
            "width":0,"height":0,
            "screenshot_path":shot,
            "context":"section=slider/promo",
        })
    return out
