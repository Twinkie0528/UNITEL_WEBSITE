# -*- coding: utf-8 -*-
# bolortoli_mn.py — Эцсийн засвар (v5)
import os
import re
import time
import hashlib
import logging
import urllib.parse
from typing import List, Dict, Set
from playwright.sync_api import sync_playwright
from common import ensure_dir, http_get_bytes, classify_ad

HOME = "https://bolor-toli.com"

def _host(u: str) -> str:
    try:
        hostname = urllib.parse.urlparse(u).hostname or ""
        return hostname[4:] if hostname.startswith('www.') else hostname
    except Exception: return ""

def _shot(output_dir: str, src: str, i: int) -> str:
    return os.path.join(output_dir, f"bolortoli_{int(time.time())}_{i}_{hashlib.md5(src.encode('utf-8','ignore')).hexdigest()[:8]}.png")

def _prime_page(page) -> None:
    for _ in range(3):
        page.mouse.wheel(0, 2500)
        page.wait_for_timeout(800)

def _collect_bolortoli(page, output_dir: str, seen: Set[str], ads_only: bool, min_score: int) -> List[Dict]:
    out: List[Dict] = []
    site_host = _host(HOME)
    
    # Carousel-ийн slide бүрийг (.v-window-item) болон бусад энгийн линкийг шалгана
    for el in page.locator("a.v-window-item, a:has(img)").all():
        try:
            # Харваас зар биш, жижиг хэмжээтэйг алгасах
            bbox = el.bounding_box()
            if not bbox or bbox['width'] < 200 or bbox['height'] < 100: continue
            w, hgt = int(bbox['width']), int(bbox['height'])

            src = ""
            # 1. CSS Background-image хайх (v-window-item-д зориулав)
            style_el = el.locator(".v-image__image").first
            if style_el.count() > 0:
                style = style_el.get_attribute("style") or ""
                match = re.search(r"url\(['\"]?(.*?)['\"]?\)", style)
                if match:
                    src = match.group(1)

            # 2. Хэрэв олдоогүй бол дотор нь img таг хайх
            if not src:
                img_el = el.locator("img").first
                if img_el.count() > 0:
                    src = img_el.get_attribute("src") or ""

            if not src or src in seen or src.startswith("data:"): continue
            src = urllib.parse.urljoin(HOME, src)
            
            landing = el.get_attribute("href") or ""
            landing = urllib.parse.urljoin(HOME, landing)

            is_ad, score, reason = classify_ad(site_host, src, landing, str(w), str(hgt), "onpage", min_score=min_score)
            if ads_only and is_ad != "1": continue
            
            seen.add(src)
            shot_path = _shot(output_dir, src, len(out))
            el.screenshot(path=shot_path)
            img_bytes = http_get_bytes(src, referer=HOME)
            out.append({"site": site_host, "src": src, "landing_url": landing, "img_bytes": img_bytes, "width": w, "height": hgt, "screenshot_path": shot_path, "notes": "onpage"})
        except Exception:
            continue
    return out

def scrape_bolortoli(output_dir: str, dwell_seconds: int = 35, headless: bool = True, ads_only: bool = True, min_score: int = 3) -> List[Dict]:
    ensure_dir(output_dir)
    seen: Set[str] = set()
    out: List[Dict] = []
    with sync_playwright() as p:
        br = p.chromium.launch(headless=headless)
        try:
            pg = br.new_page(viewport={"width": 1600, "height": 1200})
            pg.goto(HOME, timeout=90000, wait_until="networkidle")
            pg.wait_for_timeout(3000)

            _prime_page(pg)
            out.extend(_collect_bolortoli(pg, output_dir, seen, ads_only, min_score))

            if dwell_seconds > 5:
                logging.info(f"Starting active sampling for {dwell_seconds} seconds to catch rotating ads...")
                waited, step = 0, 7
                while waited < dwell_seconds:
                    time.sleep(step)
                    waited += step
                    logging.info(f"-> Resampling page... ({waited}/{dwell_seconds}s)")
                    out.extend(_collect_bolortoli(pg, output_dir, seen, ads_only, min_score))
        finally:
            br.close()

    final_out, final_seen_src = [], set()
    for item in out:
        if item['src'] not in final_seen_src:
            final_out.append(item)
            final_seen_src.add(item['src'])
    return final_out