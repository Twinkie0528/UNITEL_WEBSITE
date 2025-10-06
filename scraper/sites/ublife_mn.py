# scraper/sites/ublife_pw.py
from __future__ import annotations

import hashlib, time
from pathlib import Path
from typing import Any, Dict, List, Optional

from playwright.sync_api import sync_playwright
from common import http_get_bytes

HOME = "https://ub.life"       # ublife.mn -> ub.life руу redirect хийдэг
SITE = "ublife.mn"

DEFAULT_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/120.0.0.0 Safari/537.36")
DEFAULT_HEADERS = {"Accept-Language": "mn-MN,mn;q=0.9,en-US;q=0.6,en;q=0.5"}

MIN_W, MIN_H = 200, 100

SEL_IFRAMES = (
    # Google Ad Manager / Jegtheme slots
    "div[class*='jeg_ad'] iframe",
    "div[id^='div-gpt'] iframe",
    "ins.adsbygoogle",                # дотор нь iframe үүснэ
    # ad networks
    "iframe[src*='boost']",
    "iframe[src*='exchange']",
    "iframe[src*='doubleclick']",
    "iframe[src*='googleads']",
)

def _pick_img_src(frame) -> str:
    try:
        n = frame.locator("img[src], img[data-src]").first
        if n.count():
            return (n.get_attribute("src") or n.get_attribute("data-src") or "").strip()
    except Exception:
        pass
    try:
        s = frame.locator("source[srcset]").first
        if s.count():
            ss = (s.get_attribute("srcset") or "").split(",")[0].strip().split(" ")[0]
            return ss
    except Exception:
        pass
    try:
        v = frame.locator("video[poster]").first
        if v.count():
            return (v.get_attribute("poster") or "").strip()
    except Exception:
        pass
    return ""

def _first_href(frame) -> str:
    try:
        a = frame.locator("a[href]").first
        if a.count():
            return (a.get_attribute("href") or "").strip()
    except Exception:
        pass
    return ""

def scrape_ublife(
    output_dir: str | Path,
    *,
    dwell_seconds: int = 60,
    headless: bool = True,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    out_dir = Path(output_dir) / SITE
    out_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled",
                  "--disable-dev-shm-usage"]
        )
        context = browser.new_context(
            viewport={"width": 1600, "height": 1200},
            user_agent=DEFAULT_UA,
            locale="mn-MN",
            extra_http_headers=DEFAULT_HEADERS,
        )
        page = context.new_page()
        # stealth багахан
        page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")

        page.goto(HOME, wait_until="domcontentloaded", timeout=120000)
        try: page.wait_for_load_state("networkidle", timeout=10000)
        except: pass
        page.wait_for_timeout(5000)

        sels = ", ".join(SEL_IFRAMES)
        locator = page.locator(sels)

        # бага зэрэг доош гүйлгээд дахин шалгана
        stop_at = time.time() + dwell_seconds
        seen_keys: set[str] = set()

        def collect():
            count = locator.count()
            for i in range(count):
                el = locator.nth(i)
                try:
                    bb = el.bounding_box()
                except Exception:
                    bb = None
                if not bb: 
                    continue
                w, h = int(bb.get("width", 0)), int(bb.get("height", 0))
                if w < MIN_W or h < MIN_H:
                    # ins.adsbygoogle доторхи iframe-ийг барина
                    try:
                        inner_if = el.element_handle().query_selector("iframe")
                        if inner_if:
                            bb2 = inner_if.bounding_box()
                            if not bb2: 
                                continue
                            w, h = int(bb2["width"]), int(bb2["height"])
                            if w < MIN_W or h < MIN_H:
                                continue
                    except Exception:
                        continue

                # frame олох (ins-дотор эсвэл өөрөө iframe)
                frame = None
                try:
                    eh = el.element_handle()
                    frame = eh.content_frame()
                    if (not frame) and eh:
                        inner = eh.query_selector("iframe")
                        if inner:
                            frame = inner.content_frame()
                except Exception:
                    frame = None
                if not frame:
                    continue

                src = _pick_img_src(frame)
                if not src:
                    continue
                landing = _first_href(frame)

                key = f"{src}|{landing}|{w}x{h}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                # зураг татаж хадгална
                img_bytes = http_get_bytes(src, referer=HOME)
                fname = f"ublife_{int(time.time())}_{i}_{hashlib.sha1(src.encode('utf-8','ignore')).hexdigest()[:10]}.png"
                shot_path = str(out_dir / fname)
                try:
                    el.screenshot(path=shot_path)
                except Exception:
                    # алдаа гарвал файлгүй боловч pipeline ажиллана
                    shot_path = shot_path  # keep path for consistency

                out.append({
                    "site": SITE,
                    "src": src,
                    "landing_url": landing,
                    "landing_final_url": landing,
                    "img_bytes": img_bytes,
                    "width": w, "height": h,
                    "screenshot_path": shot_path,
                    "context": "jeg_ad/gpt",
                })

        # эхний цуглуулалт
        collect()
        while time.time() < stop_at:
            try: page.mouse.wheel(0, 1400)
            except: pass
            page.wait_for_timeout(3000)
            collect()

        context.close()
        browser.close()

    return out
