# -*- coding: utf-8 -*-
# lemonpress_mn.py — Сайжруулсан scraper for https://lemonpress.mn (v3)
import os
import time
import hashlib
import logging
import urllib.parse
from typing import List, Dict, Set
from playwright.sync_api import sync_playwright
from common import ensure_dir, http_get_bytes, classify_ad

HOME = "https://lemonpress.mn"
CAT_URL = "https://lemonpress.mn/category/surtalchilgaa"
AD_IFRAME_HINTS = ("googlesyndication.com", "doubleclick.net", "adnxs.com", "boost.mn")

def _host(u: str) -> str:
    try:
        hostname = urllib.parse.urlparse(u).hostname or ""
        return hostname[4:] if hostname.startswith('www.') else hostname
    except Exception: return ""

def _shot(output_dir: str, src: str, i: int) -> str:
    return os.path.join(output_dir, f"lemonpress_{int(time.time())}_{i}_{hashlib.md5(src.encode('utf-8','ignore')).hexdigest()[:8]}.png")

def _prime_page(page) -> None:
    for img in page.locator('img[data-src], img[data-original], img[data-srcset]').all():
        try:
            img.evaluate("(e)=>{ const setIf=(v)=>{ if(v && !e.getAttribute('src')) e.setAttribute('src', v); }; setIf(e.getAttribute('data-src')); setIf(e.getAttribute('data-original')); const ds=e.getAttribute('data-srcset'); if(ds && !e.getAttribute('src')){ const first=ds.split(',')[0].trim().split(' ')[0]; if(first)e.setAttribute('src',first); } }")
        except Exception: pass
    for _ in range(3):
        page.mouse.wheel(0, 2000)
        page.wait_for_timeout(800)

def _collect_lemonpress(page, output_dir: str, seen: Set[str], ads_only: bool, min_score: int) -> List[Dict]:
    out: List[Dict] = []
    site_host = _host(HOME)
    for el in page.locator("iframe").all():
        try:
            src = (el.get_attribute("src") or "").strip()
            if not src or src in seen or not any(hint in src for hint in AD_IFRAME_HINTS): continue
            bbox = el.bounding_box()
            if not bbox or bbox['width'] < 160 or bbox['height'] < 90: continue
            w, h = int(bbox['width']), int(bbox['height'])
            is_ad, score, reason = classify_ad(site_host, src, src, str(w), str(h), "iframe", min_score)
            if ads_only and is_ad != "1": continue
            seen.add(src)
            shot_path = _shot(output_dir, src, len(out))
            el.screenshot(path=shot_path)
            out.append({"site": site_host, "src": src, "landing_url": src, "img_bytes": b"", "width": w, "height": h, "screenshot_path": shot_path, "notes": "iframe"})
        except Exception: continue
    for el in page.locator("a:has(img)").all():
        try:
            bbox = el.bounding_box()
            if not bbox or bbox['width'] < 180 or bbox['height'] < 100: continue
            w, h = int(bbox['width']), int(bbox['height'])
            img_el = el.locator("img").first
            src = (img_el.get_attribute("src") or "").strip()
            if not src or src in seen or src.startswith("data:") or src.lower().endswith((".gif", ".svg")): continue
            landing = el.get_attribute("href") or ""
            if ads_only and not (_host(landing) and _host(landing) != site_host): continue
            is_ad, score, reason = classify_ad(site_host, src, landing, str(w), str(h), "onpage", min_score)
            if ads_only and is_ad != "1": continue
            seen.add(src)
            shot_path = _shot(output_dir, src, len(out))
            el.screenshot(path=shot_path)
            img_bytes = http_get_bytes(src, referer=page.url)
            out.append({"site": site_host, "src": src, "landing_url": landing, "img_bytes": img_bytes, "width": w, "height": h, "screenshot_path": shot_path, "notes": "onpage"})
        except Exception: continue
    return out

# --- ЗАСВАР: run.py-тэй нийцүүлэхийн тулд dwell_seconds-г нэмсэн ---
def scrape_lemonpress(output_dir: str, dwell_seconds: int = 0, headless: bool = True, ads_only: bool = True, min_score: int = 3, max_pages: int = 3) -> List[Dict]:
    ensure_dir(output_dir)
    seen: Set[str] = set()
    out: List[Dict] = []
    with sync_playwright() as p:
        br = p.chromium.launch(headless=headless)
        try:
            pg = br.new_page(viewport={"width": 1600, "height": 1200})
            logging.info("Scraping Lemonpress Homepage...")
            pg.goto(HOME, timeout=90000, wait_until="domcontentloaded")
            _prime_page(pg)
            out.extend(_collect_lemonpress(pg, output_dir, seen, ads_only, min_score))
            logging.info(f"Scraping '{CAT_URL}' category up to {max_pages} pages...")
            current_url = CAT_URL
            for i in range(max_pages):
                logging.info(f"-> Scraping page {i+1}: {current_url}")
                try:
                    pg.goto(current_url, timeout=90000, wait_until="domcontentloaded")
                    _prime_page(pg)
                    out.extend(_collect_lemonpress(pg, output_dir, seen, ads_only, min_score))
                    
                    # --- ЗАСВАР: Илүү найдвартай "Дараах" товчны сонгогч ---
                    next_link = pg.locator("a:has-text('Дараах'), a:has-text('Next'), a:has-text('»')").first
                    if next_link.count():
                        next_href = next_link.get_attribute("href")
                        if next_href:
                            current_url = urllib.parse.urljoin(HOME, next_href)
                        else: break
                    else: break
                except Exception as e:
                    logging.warning(f"Failed to process page {current_url}: {e}")
                    break
        finally:
            br.close()
            
    final_out, final_seen_src = [], set()
    for item in out:
        if item['src'] not in final_seen_src:
            final_out.append(item)
            final_seen_src.add(item['src'])
    return final_out