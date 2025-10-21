# -*- coding: utf-8 -*-
import os, time, hashlib
from typing import List, Dict, Set
from playwright.sync_api import sync_playwright
from common import ensure_dir, http_get_bytes, classify_ad

HOME = "https://news.mn"

def _shot(output_dir, src, i):
    return os.path.join(output_dir, f"news_{int(time.time())}_{i}_{hashlib.md5(src.encode('utf-8','ignore')).hexdigest()[:8]}.png")

def _collect_imgs(page, output_dir, seen:Set[str], ads_only:bool, min_score:int) -> List[Dict]:
    out: List[Dict] = []
    loc = page.locator("img, a img, div[class*='banner'] img, div[class*='ad'] img, div[id*='ad'] img, iframe, video")
    count = loc.count()
    for i in range(count):
        el = loc.nth(i)
        bbox = el.bounding_box()
        if not bbox or bbox["width"]<180 or bbox["height"]<100: continue
        tag = el.evaluate("e => e.tagName.toLowerCase()")
        src = ""
        if tag == "video":
            src = el.get_attribute("poster") or ""
        else:
            src = el.get_attribute("src") or ""
        if not src or src.startswith("data:"): continue
        if src.lower().endswith(".gif"): continue
        if src in seen: continue
        seen.add(src)

        landing = ""
        try:
            parent = el.evaluate_handle("e => e.closest('a')")
            landing = parent.get_property("href").json_value() if parent else ""
        except Exception: pass
        if not landing: landing = HOME

        is_ad, score, reason = classify_ad("news.mn", src, landing, str(int(bbox["width"])), str(int(bbox["height"])), ("iframe" if tag=="iframe" else "onpage"), min_score=min_score)
        if ads_only and is_ad != "1":
            continue

        img_bytes = http_get_bytes(src, referer=HOME)
        shot = _shot(output_dir, src, i)
        try: el.screenshot(path=shot)
        except Exception: shot = ""
        out.append({
            "site":"news.mn","src":src,"landing_url":landing,
            "img_bytes":img_bytes,"width":int(bbox["width"]),"height":int(bbox["height"]),
            "screenshot_path":shot,"notes":("video_poster" if tag=="video" else ("iframe" if tag=="iframe" else "onpage")),
        })
    return out

def scrape_news(output_dir: str, dwell_seconds:int=45, headless:bool=True, ads_only:bool=True, min_score:int=2) -> List[Dict]:
    ensure_dir(output_dir)
    seen: Set[str] = set(); out: List[Dict] = []
    with sync_playwright() as p:
        br = p.chromium.launch(headless=headless)
        pg = br.new_page(viewport={"width":1600,"height":1200})
        pg.goto(HOME, timeout=90000, wait_until="domcontentloaded")
        out += _collect_imgs(pg, output_dir, seen, ads_only, min_score)
        waited, step = 0, 6
        while waited < dwell_seconds:
            time.sleep(step); waited += step
            try: out += _collect_imgs(pg, output_dir, seen, ads_only, min_score)
            except Exception: pass
        br.close()
    return out