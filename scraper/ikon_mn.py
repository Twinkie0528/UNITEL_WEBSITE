from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin
from textwrap import dedent

from playwright.sync_api import ElementHandle, Page, sync_playwright

from common import http_get_bytes

HOME = "https://ikon.mn"
BLOCK_PATTERNS = (
    "thumb",
    "thumbnail",
    "video-thumb",
    "story-card",
    "avatar",
    "logo",
    "icon",
    "cover",
    "/thumb/",
    "/cache/",
    "/icons/",
)

_LANDING_ATTRS = [
    "data-href",
    "data-url",
    "data-link",
    "data-target",
    "data-dest",
    "data-redirect",
    "data-out",
]

_JS_EXTRACT_SRC = dedent(
    """
    (node) => {
      const pick = (value) => {
        if (!value) return '';
        const clean = value.trim();
        if (clean.includes(',')) {
          return clean.split(',')[0].trim().split(' ')[0];
        }
        return clean;
      };
      const lower = node.tagName.toLowerCase();
      if (lower === 'img') {
        const direct = node.currentSrc || node.src || node.getAttribute('data-src') || node.getAttribute('data-original') || node.getAttribute('data-lazy-src') || node.getAttribute('data-source') || node.getAttribute('data-url');
        if (direct) return pick(direct);
        const srcset = node.getAttribute('srcset') || node.getAttribute('data-srcset');
        if (srcset) return pick(srcset);
      }
      if (lower === 'video') {
        return node.getAttribute('poster') || '';
      }
      if (lower === 'source') {
        const srcset = node.getAttribute('srcset') || node.getAttribute('data-srcset');
        if (srcset) return pick(srcset);
      }
      const attrSrc = node.getAttribute('src') || node.getAttribute('data-src') || '';
      return pick(attrSrc);
    }
    """
)


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    url = url.strip()
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/") and not url.startswith("//"):
        return urljoin(HOME, url)
    return url


def _extract_src(el: ElementHandle) -> str:
    try:
        src = el.evaluate(_JS_EXTRACT_SRC)
        return _normalize_url(src or "")
    except Exception:
        return ""


def _extract_landing(el: ElementHandle) -> str:
    landing = ""
    try:
        parent = el.evaluate_handle("e => e.closest('a')")
        if parent:
            landing = parent.get_property("href").json_value()
    except Exception:
        landing = ""
    if landing:
        return _normalize_url(landing)
    for attr in _LANDING_ATTRS:
        try:
            candidate = el.get_attribute(attr) or ""
        except Exception:
            candidate = ""
        if candidate:
            return _normalize_url(candidate)
    return ""


def _extract_iframe_details(el: ElementHandle) -> Dict[str, Optional[str]]:
    data: Dict[str, Optional[str]] = {"src": None, "landing": None}
    try:
        frame = el.content_frame()
    except Exception:
        frame = None
    if not frame:
        return data
    try:
        frame.wait_for_load_state("domcontentloaded", timeout=3000)
    except Exception:
        pass
    try:
        anchor = frame.query_selector("a[href]")
        if anchor:
            link = anchor.get_attribute("href")
            if link:
                data["landing"] = _normalize_url(link)
    except Exception:
        pass
    try:
        inner_img = frame.query_selector("img[src]")
        if inner_img:
            data["src"] = _normalize_url(inner_img.get_attribute("src") or "")
        else:
            video = frame.query_selector("video")
            if video:
                poster = video.get_attribute("poster") or video.get_attribute("src")
                if poster:
                    data["src"] = _normalize_url(poster)
    except Exception:
        pass
    return data


def _shot(output_dir: Path, src: str, index: int) -> str:
    fname = f"ikon_{int(time.time())}_{index}_{hashlib.md5(src.encode('utf-8', 'ignore')).hexdigest()[:8]}.png"
    return str(output_dir / fname)


def _should_skip(src: str, classes: str) -> bool:
    value = (src or "").lower()
    cls = (classes or "").lower()
    for token in BLOCK_PATTERNS:
        if token in value or token in cls:
            return True
    return False


def _collect_candidates(page: Page, output_dir: Path, seen: Set[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    locator = page.locator(
        "div[id*='banner'] img, div[class*='banner'] img, div[class*='ads'] img, div[id*='ads'] img, "
        "section[class*='sponsor'] img, aside[class*='banner'] img, iframe[src*='ads'], iframe[id*='ad'], iframe[class*='ad']"
    )
    count = locator.count()
    for i in range(count):
        el = locator.nth(i)
        try:
            bbox = el.bounding_box()
        except Exception:
            bbox = None
        if not bbox or bbox["width"] < 180 or bbox["height"] < 100:
            continue

        classes = el.get_attribute("class") or ""
        tag = el.evaluate("node => node.tagName.toLowerCase()")
        src = _extract_src(el)
        if not src or src.startswith("data:"):
            continue
        if _should_skip(src, classes):
            continue
        if src.lower().endswith((".gif", ".webp")):
            continue

        iframe_meta: Optional[Dict[str, Optional[str]]] = None
        if tag == "iframe":
            iframe_meta = _extract_iframe_details(el)
            if iframe_meta.get("src"):
                src = iframe_meta.get("src") or src

        if src in seen:
            continue
        seen.add(src)

        landing = _extract_landing(el)
        if iframe_meta and not landing:
            landing = iframe_meta.get("landing") or ""

        img_bytes = http_get_bytes(src, referer=HOME)
        shot_path = _shot(output_dir, src, i)
        try:
            el.screenshot(path=shot_path)
        except Exception:
            shot_path = ""

        out.append(
            {
                "site": "ikon.mn",
                "src": src,
                "landing_url": landing,
                "img_bytes": img_bytes,
                "width": int(bbox["width"]),
                "height": int(bbox["height"]),
                "screenshot_path": shot_path,
                "context": f"classes={classes} tag={tag}",
            }
        )
    return out


def scrape_ikon(
    output_dir: str | Path,
    *,
    dwell_seconds: int = 55,
    headless: bool = True,
) -> List[Dict[str, Any]]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: Set[str] = set()
    results: List[Dict[str, Any]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page(viewport={"width": 1600, "height": 1200})
        page.goto(HOME, timeout=90000, wait_until="domcontentloaded")
        results.extend(_collect_candidates(page, out_dir, seen))
        waited = 0
        step = 6
        while waited < dwell_seconds:
            time.sleep(step)
            waited += step
            try:
                results.extend(_collect_candidates(page, out_dir, seen))
            except Exception:
                pass
        browser.close()
    return results
