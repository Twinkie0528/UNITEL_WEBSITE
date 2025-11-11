# -*- coding: utf-8 -*-
# caak_mn.py — FAST+FOCUSED scraper for https://www.caak.mn/ (САЙЖРУУЛСАН ХУВИЛБАР)
import os
import time
import hashlib
import urllib.parse
from typing import List, Dict, Set
# <<< ЗАСВАР 1: Алдаа барихын тулд 'Error'-г 'PlaywrightError' нэрээр импортлох
from playwright.sync_api import sync_playwright, Error as PlaywrightError
from common import ensure_dir, http_get_bytes, classify_ad

HOME = "https://www.caak.mn/"
# <<< ЗАСВАР 2: Жинхэнэ хөтөч мэт харагдуулах User-Agent нэмэх
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"


AD_IFRAME_HINTS = (
    "googlesyndication.com", "doubleclick.net", "adnxs.com", "ads.pubmatic.com",
    "rubiconproject.com", "criteo.net", "taboola.com", "outbrain.com"
)
ACTIVE_STORAGE_HINT = "/active_storage/representations/redirect/"

def _host(u: str) -> str:
    """URL-аас домайн нэрийг салгаж авах функц."""
    try:
        hostname = urllib.parse.urlparse(u).hostname or ""
        # 'www.'-г хасах
        if hostname.startswith('www.'):
            return hostname[4:]
        return hostname
    except Exception:
        return ""

def _shot(output_dir: str, src: str, i: int) -> str:
    """Screenshot хадгалах файлын замыг үүсгэх."""
    return os.path.join(
        output_dir,
        f"caak_{int(time.time())}_{i}_{hashlib.md5(src.encode('utf-8','ignore')).hexdigest()[:8]}.png"
    )

def _prime_page(page) -> None:
    """Хуудсыг бэлдэх: lazy-load-ыг өдөөх, слайдер гүйлгэх, доош скроллох."""
    page.evaluate("document.querySelectorAll('img[loading=\"lazy\"]').forEach(img => img.loading = 'eager')")
    for _ in range(3):
        page.mouse.wheel(0, 2500)
        page.wait_for_timeout(800)

    sels = [".slick-next", ".swiper-button-next", "[data-carousel-next]", ".owl-next"]
    for _ in range(2):
        for sel in sels:
            try:
                el = page.locator(sel).first
                if el and el.count() and el.is_visible():
                    el.click(timeout=1000)
                    page.wait_for_timeout(450)
            except Exception:
                pass

def _collect_caak(page, output_dir: str, seen: Set[str], ads_only: bool, min_score: int) -> List[Dict]:
    """Хуудаснаас зар сурталчилгааны элементүүдийг цуглуулах үндсэн функц."""
    out: List[Dict] = []
    site_host = _host(HOME)

    # 1) IFRAME ЗАРЫГ ШАЛГАХ (Илүү найдвартай)
    for el in page.locator("iframe").all():
        src = (el.get_attribute("src") or "").strip()
        if not src or src in seen or src.startswith("data:"):
            continue

        is_ad_domain = any(hint in src for hint in AD_IFRAME_HINTS)
        if ads_only and not is_ad_domain:
            continue

        try:
            bbox = el.bounding_box()
            if not bbox or bbox['width'] < 160 or bbox['height'] < 90:
                continue
            w, hgt = int(bbox['width']), int(bbox['height'])

            # Iframe нь ихэвчлэн өөртөө landing URL агуулдаггүй
            landing = src
            notes = "iframe"
            is_ad, score, reason = classify_ad(site_host, src, landing, str(w), str(hgt), notes, min_score=min_score)

            if ads_only and is_ad != "1":
                continue

            shot_path = _shot(output_dir, src, len(out))
            el.screenshot(path=shot_path)
            
            out.append({
                "site": site_host, "src": src, "landing_url": landing,
                "img_bytes": b"", "width": w, "height": hgt,
                "screenshot_path": shot_path, "notes": notes,
            })
            seen.add(src)
        except Exception:
            continue

    # 2) IMG ЗАРЫГ ШАЛГАХ (Илүү нарийн шүүлтүүр хийнэ)
    for el in page.locator("img, a img").all():
        try:
            bbox = el.bounding_box()
            if not bbox or bbox['width'] < 180 or bbox['height'] < 100:
                continue
            w, hgt = int(bbox['width']), int(bbox['height'])

            src = (el.get_attribute("src") or "").strip()
            if not src or src in seen or src.startswith("data:") or src.lower().endswith((".gif", ".svg")):
                continue

            # --- САЙЖРУУЛАЛТ: Холбоосыг шалгаж, зөвхөн ГАДААД холбоостойг авч үлдэнэ ---
            landing = ""
            parent_a = el.locator("xpath=ancestor::a").first
            if parent_a:
                landing = parent_a.get_attribute("href") or ""

            landing_host = _host(landing)
            is_external = landing_host and landing_host != site_host

            # Хэрэв `ads_only` бол зөвхөн гадаад холбоостойг л сурталчилгааны кандидат гэж үзнэ.
            if ads_only and not is_external:
                continue

            # classify_ad функцээр эцсийн дүгнэлт хийлгэх
            notes = "onpage_active_storage" if ACTIVE_STORAGE_HINT in src else "onpage"
            is_ad, score, reason = classify_ad(site_host, src, landing, str(w), str(hgt), notes, min_score=min_score)

            if ads_only and is_ad != "1":
                continue

            shot_path = _shot(output_dir, src, len(out))
            el.screenshot(path=shot_path)
            img_bytes = http_get_bytes(src, referer=HOME)
            
            out.append({
                "site": site_host, "src": src, "landing_url": landing,
                "img_bytes": img_bytes, "width": w, "height": hgt,
                "screenshot_path": shot_path, "notes": notes,
            })
            seen.add(src)
        except Exception:
            continue

    return out

def scrape_caak(output_dir: str, dwell_seconds: int = 45, headless: bool = True,
                ads_only: bool = True, min_score: int = 5) -> List[Dict]:
    """
    caak.mn сайтаас зар сурталчилгааг илрүүлж, screenshot хийдэг сайжруулсан scraper.
    
    Гол онцлог:
      - `min_score=5`: Зар гэж таних онооны босгыг өндөрсгөсөн.
      - `ads_only=True`: Зөвхөн зар гэж ангилагдсан, гадаад холбоостой зургуудыг шүүнэ.
    
    Ашиглалт:
        from caak_mn import scrape_caak
        items = scrape_caak("./screenshots", headless=True)
    """
    ensure_dir(output_dir)
    seen: Set[str] = set()
    out: List[Dict] = []

    with sync_playwright() as p:
        br = p.chromium.launch(headless=headless)
        try:
            # <<< ЗАСВАР 3: Browser Context үүсгэж, User-Agent-г тохируулах
            context = br.new_context(
                user_agent=USER_AGENT,
                viewport={"width": 1680, "height": 1200}
            )
            context.set_default_navigation_timeout(90000) # Timeout-г context-д тохируулах
            pg = context.new_page()

            # <<< ЗАСВАР 4: Алдаа гарвал дахин оролдох (Retry) логик
            MAX_RETRIES = 3
            for attempt in range(MAX_RETRIES):
                try:
                    print(f"INFO: {HOME}-д хандаж байна (Оролдлого {attempt + 1}/{MAX_RETRIES})...")
                    pg.goto(HOME, wait_until="domcontentloaded")
                    print(f"INFO: {HOME}-д амжилттай хандлаа.")
                    break # Амжилттай бол давталтаас гарна
                except PlaywrightError as e:
                    print(f"WARNING: {HOME}-д хандах үед алдаа гарлаа (Оролдлого {attempt + 1}): {e}")
                    if attempt < MAX_RETRIES - 1:
                        wait_time = (attempt + 1) * 5 # 5, 10 секунд хүлээх
                        print(f"INFO: {wait_time} секунд хүлээгээд дахин оролдоно...")
                        time.sleep(wait_time)
                    else:
                        print(f"FATAL: {MAX_RETRIES} удаа оролдоод {HOME}-д хандаж чадсангүй.")
                        raise e # Хамгийн сүүлийн алдааг дээш шидэх
            
            # --- Эндээс цааш код хэвийн үргэлжилнэ ---

            _prime_page(pg)

            out.extend(_collect_caak(pg, output_dir, seen, ads_only, min_score))

            # Нэмэлт зар гарч ирэхийг хүлээх (Dwell time)
            if dwell_seconds > 0:
                time.sleep(dwell_seconds)
                _prime_page(pg)
                out.extend(_collect_caak(pg, output_dir, seen, ads_only, min_score))
        finally:
            br.close()
            
    return out