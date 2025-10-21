# -*- coding: utf-8 -*-
# ublife_mn.py — FAST scraper for https://www.ublife.mn/
import os, time, hashlib
from typing import List, Dict, Set
from playwright.sync_api import sync_playwright
from common import ensure_dir, http_get_bytes, classify_ad

HOME = "https://ublife.mn/"

def _shot(output_dir: str, src: str, i: int) -> str:
    return os.path.join(
        output_dir,
        f"ublife_{int(time.time())}_{i}_{hashlib.md5(src.encode('utf-8','ignore')).hexdigest()[:8]}.png"
    )

def _collect_imgs(page, output_dir: str, seen: Set[str], ads_only: bool, min_score: int) -> List[Dict]:
    out: List[Dict] = []
    loc = page.locator(
        "img, a img, "
        "div[class*='banner'] img, div[class*='ad'] img, div[id*='ad'] img, "
        "aside[class*='ad'] img, section[class*='ad'] img, "
        "iframe, video"
    )
    count = loc.count()
    for i in range(count):
        el = loc.nth(i)

        # Skip small
        try:
            bbox = el.bounding_box()
        except Exception:
            bbox = None
        if not bbox:
            continue
        w, h = int(bbox.get("width", 0)), int(bbox.get("height", 0))
        if w < 180 or h < 100:
            continue

        # Tag
        try:
            tag = el.evaluate("e => e.tagName.toLowerCase()")
        except Exception:
            tag = ""

        # SRC / POSTER
        try:
            src = el.get_attribute("poster") if tag == "video" else el.get_attribute("src")
            src = src or ""
        except Exception:
            src = ""
        if not src or src.startswith("data:"):
            continue
        if src.lower().endswith(".gif"):
            continue
        if src in seen:
            continue
        seen.add(src)

        # Landing (closest <a>)
        landing = ""
        try:
            parent = el.evaluate_handle("e => e.closest('a')")
            landing = parent.get_property("href").json_value() if parent else ""
        except Exception:
            pass
        if not landing:
            landing = HOME

        # Classify
        notes = "video_poster" if tag == "video" else ("iframe" if tag == "iframe" else "onpage")
        is_ad, score, reason = classify_ad("ublife.mn", src, landing, str(w), str(h), notes, min_score=min_score)
        if ads_only and is_ad != "1":
            continue

        # Bytes + screenshot
        img_bytes = http_get_bytes(src, referer=HOME)
        shot = _shot(output_dir, src, i)
        try:
            el.screenshot(path=shot)
        except Exception:
            shot = ""

        out.append({
            "site": "ublife.mn",
            "src": src,
            "landing_url": landing,
            "img_bytes": img_bytes,
            "width": w,
            "height": h,
            "screenshot_path": shot,
            "notes": notes,
        })
    return out

def scrape_ublife(output_dir: str, dwell_seconds: int = 45, headless: bool = True,
                  ads_only: bool = True, min_score: int = 3) -> List[Dict]:
    """
    Ашиглах жишээ:
        from ublife_mn import scrape_ublife
        items = scrape_ublife("./banner_screenshots/2025-10-10",
                              dwell_seconds=45, headless=True, ads_only=True, min_score=3)
    """
    ensure_dir(output_dir)
    seen: Set[str] = set()
    out: List[Dict] = []

    with sync_playwright() as p:
        br = p.chromium.launch(headless=headless)
        pg = br.new_page(viewport={"width": 1600, "height": 1200})
        pg.goto(HOME, timeout=90000, wait_until="domcontentloaded")

        # Initial grab
        out += _collect_imgs(pg, output_dir, seen, ads_only, min_score)

        # Short dwell loop
        waited, step = 0, 6
        while waited < dwell_seconds:
            time.sleep(step); waited += step
            try:
                out += _collect_imgs(pg, output_dir, seen, ads_only, min_score)
            except Exception:
                pass

        br.close()
    return out
