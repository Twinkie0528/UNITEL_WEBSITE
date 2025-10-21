# -*- coding: utf-8 -*-
# ikon_mn.py — Шинэ, хоёр шатлалт логикоор шинэчилсэн хувилбар
import os
import re
import time
import hashlib
import logging
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Set, Optional, Tuple

from playwright.sync_api import sync_playwright, Page, BrowserContext, TimeoutError as PWTimeout
from common import ensure_dir, http_get_bytes # Таны common.py-аас импорт хийнэ

HOME = "https://ikon.mn"
AD_PATH_HINT = "/ad/"

# ---- Timing / Filters ----
# run.py-аас ирэх dwell_seconds-г ашиглана
RELOAD_ROUNDS = 2 # Нэг /ad/ хуудсыг хэдэн удаа дахин ачааллах
MIN_W, MIN_H = 100, 100

def _shot(output_dir: str, src: str, i: int) -> str:
    """Screenshot-ын замыг үүсгэх."""
    return os.path.join(output_dir, f"ikon_{int(time.time())}_{i}_{hashlib.md5(src.encode('utf-8','ignore')).hexdigest()[:8]}.png")

def _host(u: str) -> str:
    """URL-аас host-ийг салгах."""
    try: return urlparse(u).netloc.lower()
    except Exception: return ""

# --- 1-Р ШАТ: /ad/ ХОЛБООСУУДЫГ ОЛОХ ХЭРЭГСЛҮҮД ---
def _collect_ad_links_dom(page: Page) -> Set[str]:
    """HTML-ээс /ad/ агуулсан a болон iframe холбоосыг олно."""
    ad_links: Set[str] = set()
    selectors = "iframe[src*='/ad/'], a[href*='/ad/']"
    for el in page.locator(selectors).all():
        try:
            attr = "src" if el.evaluate("e => e.tagName") == 'IFRAME' else "href"
            link = el.get_attribute(attr) or ""
            if AD_PATH_HINT in link:
                ad_links.add(urljoin(HOME, link))
        except Exception:
            pass
    return ad_links

def _collect_ad_links_network(context: BrowserContext, page: Page, settle_seconds: int) -> Set[str]:
    """Сүлжээний хүсэлтүүдийг чагнаж /ad/ холбоосыг олно."""
    ad_links: Set[str] = set()
    def maybe_add(url: str):
        if AD_PATH_HINT in url and "ikon.mn" in _host(url):
            ad_links.add(url)

    context.on("request", lambda req: maybe_add(req.url))
    
    t_end = time.time() + settle_seconds
    while time.time() < t_end:
        page.mouse.wheel(0, 1200)
        page.wait_for_timeout(250)
    return ad_links

def find_ad_links_on_home(context: BrowserContext, page: Page, idle_seconds: int) -> List[str]:
    """Бүх аргаар /ad/ холбоосуудыг олж, нэгтгэнэ."""
    collected: Set[str] = set()
    collected.update(_collect_ad_links_dom(page))
    collected.update(_collect_ad_links_network(context, page, idle_seconds))
    return sorted([link for link in collected if "ikon.mn" in _host(link) and AD_PATH_HINT in link])


# --- 2-Р ШАТ: /ad/ ХУУДАС БҮРИЙГ "АЖИГЛАЖ" ЗАРЫГ БАРЬЖ АВАХ ---
def _guess_click_url(page: Page) -> str:
    """Хуудсан дээрх гадаад холбоосыг таамаглан олно."""
    try:
        page_host = _host(page.url)
        for a in page.locator("a[href^='http']").all():
            href = a.get_attribute("href") or ""
            href_host = _host(href)
            if href_host and href_host != page_host:
                return href # Эхний гадаад холбоосыг буцаана
    except Exception:
        pass
    return ""

def watch_and_capture_variants(context: BrowserContext, ad_url: str, output_dir: str, seen: Set[str], total_watch_seconds: int) -> List[Dict]:
    """Нэг /ad/ хуудсыг ажиглаж, гарч ирсэн бүх зарын хувилбарыг барьж авна."""
    captures: List[Dict] = []
    
    page = context.new_page()
    try:
        page.set_default_timeout(15000)
        round_seconds = max(5, total_watch_seconds // RELOAD_ROUNDS)

        for _ in range(RELOAD_ROUNDS):
            page.goto(ad_url, wait_until="domcontentloaded")
            
            t_end = time.time() + round_seconds
            while time.time() < t_end:
                for el in page.locator("img[data-banner-target='item']").all():
                    try:
                        if not el.is_visible(): continue

                        src = el.get_attribute("src") or ""
                        if not src or src in seen or src.startswith("data:"): continue
                        
                        abs_url = urljoin(ad_url, src)
                        if abs_url in seen: continue
                        seen.add(abs_url)

                        bbox = el.bounding_box()
                        w = int(bbox.get("width", 0)) if bbox else 0
                        h = int(bbox.get("height", 0)) if bbox else 0

                        if w < MIN_W or h < MIN_H: continue

                        img_bytes = http_get_bytes(abs_url, referer=ad_url)
                        if not img_bytes: continue
                        
                        shot_path = _shot(output_dir, abs_url, len(captures) + 1)
                        el.screenshot(path=shot_path)
                        
                        click_url = _guess_click_url(page)

                        logging.info(f"[ikon.mn] Captured new ad creative: {abs_url}")
                        captures.append({
                            "site": "ikon.mn",
                            "src": abs_url,
                            "landing_url": click_url,
                            "img_bytes": img_bytes,
                            "width": w,
                            "height": h,
                            "screenshot_path": shot_path,
                            "notes": f"from_ad_page:{ad_url}",
                        })
                    except Exception:
                        continue
                page.wait_for_timeout(2000) # 2 секунд тутамд дахин шалгана
    except Exception as e:
        logging.warning(f"Failed to watch ad page {ad_url}: {e}")
    finally:
        page.close()

    return captures

# --- ҮНДСЭН ФУНКЦ ---
def scrape_ikon(output_dir: str, dwell_seconds: int = 45, headless: bool = True, ads_only: bool = True, min_score: int = 3) -> List[Dict]:
    ensure_dir(output_dir)
    seen: Set[str] = set()
    results: List[Dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()
        try:
            page.goto(HOME, wait_until="networkidle")
            
            # 1-р шат: /ad/ холбоосуудыг олох
            homepage_idle_seconds = max(5, dwell_seconds // 3)
            ad_links = find_ad_links_on_home(context, page, homepage_idle_seconds)
            logging.info(f"Found {len(ad_links)} unique /ad/ page links on ikon.mn homepage.")

            # 2-р шат: /ad/ хуудас бүрийг ажиглах
            if ad_links:
                watch_seconds_per_link = max(10, (dwell_seconds * 2) // (3 * len(ad_links)))
                for ad_url in ad_links:
                    logging.info(f"-> Watching ad page: {ad_url} for ~{watch_seconds_per_link}s...")
                    captures = watch_and_capture_variants(context, ad_url, output_dir, seen, watch_seconds_per_link)
                    results.extend(captures)

        except Exception as e:
            logging.error(f"An error occurred during ikon.mn scrape: {e}")
        finally:
            context.close()
            browser.close()

    # Давхардлыг эцсийн байдлаар шүүх
    final_out: List[Dict] = []
    final_seen_src: Set[str] = set()
    for item in results:
        if item.get("src") and item["src"] not in final_seen_src:
            final_out.append(item)
            final_seen_src.add(item["src"])

    logging.info(f"Completed ikon.mn scrape, found {len(final_out)} new unique creatives.")
    return final_out