from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup, Tag

from common import BLOCKED_TOKENS, canonical_src, final_redirect_url, http_get, http_get_bytes
from .utils import absolutize, extract_img_src, save_image_bytes

logger = logging.getLogger(__name__)

SITE = "ikon.mn"
HOME = "https://ikon.mn"

IMAGE_SELECTORS = (
    "div[data-controller='banner'] img",
    "div[data-controller='banner'] video[poster]",
    "div[id^='IKONMN_'] img",
    "div[id*='banner'] img",
    "div[class*='banner'] img",
    "div[id*='ad'] img",
    "div[class*='ad'] img",
    "section[class*='sponsor'] img",
)

IFRAME_KEYWORDS = (
    "boost",
    "exchange",
    "googlesyndication",
    "doubleclick",
    "taboola",
    "mgid",
    "criteo",
    "adnxs",
)


def scrape_ikon(output_dir: Path) -> List[Dict[str, Any]]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    captures: List[Dict[str, Any]] = []
    seen: set[str] = set()

    try:
        response = http_get(HOME)
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("ikon.mn fetch failed: %s", exc)
        return captures

    soup = BeautifulSoup(response.text, "lxml")
    captures.extend(
        _extract_assets(soup, HOME, out_dir, seen, starting_index=len(captures))
    )
    return captures


def _extract_assets(
    soup: BeautifulSoup,
    page_url: str,
    out_dir: Path,
    seen: set[str],
    *,
    starting_index: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    selector = ",".join(IMAGE_SELECTORS)
    for img in soup.select(selector):
        capture = _capture_image(img, page_url, out_dir, seen, index=starting_index + len(results))
        if capture:
            results.append(capture)

    for iframe in soup.select("iframe[src]"):
        capture = _capture_iframe(iframe, page_url, seen)
        if capture:
            results.append(capture)

    return results


def _capture_image(
    element: Tag,
    page_url: str,
    out_dir: Path,
    seen: set[str],
    *,
    index: int,
) -> Optional[Dict[str, Any]]:
    raw_src = extract_img_src(element)
    abs_src = absolutize(raw_src, page_url)
    if not abs_src:
        return None
    src_lower = abs_src.lower()
    if any(token in src_lower for token in BLOCKED_TOKENS):
        return None
    src_key = canonical_src(abs_src)
    if src_key and src_key in seen:
        return None

    parent_anchor = element.find_parent("a")
    landing = absolutize(parent_anchor.get("href") if parent_anchor else "", page_url)
    landing = landing or page_url

    img_bytes = http_get_bytes(abs_src, referer=page_url)
    screenshot_path = ""
    width = int(element.get("width") or 0)
    height = int(element.get("height") or 0)
    if img_bytes:
        saved_path, saved_w, saved_h = save_image_bytes(img_bytes, out_dir, SITE, abs_src, index)
        if saved_path:
            screenshot_path = saved_path
            width = saved_w or width
            height = saved_h or height
    else:
        img_bytes = None

    landing_final = final_redirect_url(landing, referer=page_url) if landing else ""

    classes = " ".join(element.get("class") or [])
    element_id = element.get("id") or ""
    context = f"tag={element.name} classes={classes} id={element_id}"

    if src_key:
        seen.add(src_key)

    return {
        "site": SITE,
        "src": abs_src,
        "landing_url": landing,
        "landing_final_url": landing_final,
        "img_bytes": img_bytes,
        "width": width,
        "height": height,
        "screenshot_path": screenshot_path,
        "context": context,
    }


def _capture_iframe(iframe: Tag, page_url: str, seen: set[str]) -> Optional[Dict[str, Any]]:
    raw_src = iframe.get("src")
    abs_src = absolutize(raw_src, page_url)
    if not abs_src:
        return None
    if not any(keyword in abs_src for keyword in IFRAME_KEYWORDS):
        return None
    src_key = canonical_src(abs_src)
    if src_key and src_key in seen:
        return None

    landing = abs_src
    landing_final = final_redirect_url(landing, referer=page_url)
    width = int(iframe.get("width") or 0)
    height = int(iframe.get("height") or 0)
    classes = " ".join(iframe.get("class") or [])
    element_id = iframe.get("id") or ""
    context = f"tag=iframe classes={classes} id={element_id}"

    if src_key:
        seen.add(src_key)

    return {
        "site": SITE,
        "src": abs_src,
        "landing_url": landing,
        "landing_final_url": landing_final,
        "img_bytes": None,
        "width": width,
        "height": height,
        "screenshot_path": "",
        "context": context,
    }
