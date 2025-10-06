from __future__ import annotations

import hashlib
import io
import logging
import time
import re
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag
from PIL import Image

logger = logging.getLogger(__name__)


def norm(url: str, base: str) -> str:
    if not url: return ""
    url = url.strip()
    if url.startswith("//"): return "https:" + url
    if url.startswith("/") and not url.startswith("//"): return urljoin(base, url)
    return url

IMG_ATTRS = ("src", "data-src", "data-original", "data-lazy", "data-url")

def pick_src_from_img(img, base: str) -> str:
    if not img: return ""
    # srcset first candidate
    if img.has_attr("srcset") and img["srcset"]:
        cand = img["srcset"].split(",")[0].strip().split(" ")[0]
        return norm(cand, base)
    for a in IMG_ATTRS:
        if img.has_attr(a) and img[a]:
            return norm(img[a], base)
    return ""

_BG_URL = re.compile(r"url\((['\"]?)(.+?)\1\)", re.I)
def pick_bg_from_style(style: str, base: str) -> str:
    if not style: return ""
    m = _BG_URL.search(style)
    return norm(m.group(2), base) if m else ""

def first_external_link(soup, publisher_host: str) -> str:
    for a in soup.select("a[href^='http']"):
        try:
            host = urlparse(a["href"]).netloc.lower()
        except Exception:
            continue
        if host and publisher_host not in host:
            return a["href"]
    return ""



def absolutize(url: Optional[str], base: str) -> str:
    if not url:
        return ""
    url = url.strip()
    if not url:
        return ""
    if url.startswith("//"):
        return f"https:{url}"
    if url.startswith("/"):
        return urljoin(base, url)
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return urljoin(base, url)


def extract_img_src(node: Tag) -> str:
    if not node:
        return ""
    candidates: Iterable[Optional[str]] = (
        node.get("data-src"),
        node.get("data-original"),
        node.get("data-lazy-src"),
        node.get("data-srcset"),
        node.get("srcset"),
        node.get("src"),
        node.get("data-url"),
    )
    for raw in candidates:
        if not raw:
            continue
        raw = raw.strip()
        if not raw:
            continue
        if "," in raw:
            raw = raw.split(",", 1)[0].strip().split(" ")[0]
        if raw:
            return raw
    return ""


def first_external_link(soup: BeautifulSoup, allowed_hosts: Iterable[str]) -> str:
    hosts = tuple(h.lower() for h in allowed_hosts)
    for anchor in soup.select("a[href^='http']"):
        href = anchor.get("href")
        if not href:
            continue
        parsed = urlparse(href)
        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if any(host.endswith(allowed) for allowed in hosts):
            continue
        return href
    return ""


def save_image_bytes(
    img_bytes: bytes,
    output_dir: Path,
    site_slug: str,
    key: str,
    index: int,
) -> tuple[str, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        image = Image.open(io.BytesIO(img_bytes))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Image open failed for %s: %s", key, exc)
        return "", 0, 0

    try:
        image = image.convert("RGB")
    except Exception:
        pass
    width, height = image.size
    digest = hashlib.sha1((key or "").encode("utf-8", "ignore") + img_bytes).hexdigest()[:10]
    filename = f"{site_slug.replace('.', '_')}_{int(time.time())}_{index:03d}_{digest}.png"
    path = output_dir / filename
    try:
        image.save(path, format="PNG")
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to save image %s: %s", path, exc)
        return "", 0, 0
    return str(path), width, height
