from __future__ import annotations

import hashlib
import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import imagehash
import requests
from PIL import Image

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

AD_KEYWORDS: Tuple[str, ...] = (
    "ad",
    "ads",
    "promo",
    "banner",
    "sponsor",
    "зар",
    "сурталчилгаа",
    "surtalchilgaa",
    "advertorial",
    "sponsored",
)

BLOCKED_TOKENS: Tuple[str, ...] = (
    "appstore",
    "playstore",
    "app-store",
    "badge",
    "qr",
    "sprite",
    "footer",
    "social",
    "logo",
    "icon",
    "thumbnail",
    "thumb",
    "avatar",
)

KNOWN_AD_HOSTS: Tuple[str, ...] = (
    "boost.mn",
    "exchange.boost.mn",
    "googlesyndication.com",
    "doubleclick.net",
    "taboola.com",
    "mgid.com",
    "criteo.com",
    "adnxs.com",
)

STANDARD_SIZES: Tuple[Tuple[int, int], ...] = (
    (300, 250),
    (300, 600),
    (320, 50),
    (320, 100),
    (336, 280),
    (468, 60),
    (728, 90),
    (970, 90),
    (970, 250),
    (1080, 1920),
)

MAX_IMAGE_BYTES = 3_000_000


def normalize_site_host(site: str) -> str:
    if not site:
        return ""
    site = site.strip()
    if not site:
        return ""
    if "//" not in site:
        site = f"https://{site}"
    try:
        host = urlparse(site).netloc.lower()
        return host
    except Exception:
        return site.lower()


def host_from_url(url: str) -> str:
    if not url:
        return ""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def canonical_src(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url)
    except Exception:
        return url.lower()
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = (parsed.path or "").rstrip("/")
    return f"{host}{path}"


def http_get(url: str, *, referer: Optional[str] = None, timeout: int = 15) -> requests.Response:
    headers = {"User-Agent": USER_AGENT}
    if referer:
        headers["Referer"] = referer
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response


def http_get_bytes(
    url: str,
    *,
    referer: Optional[str] = None,
    timeout: int = 15,
    allow_webp_gif: bool = True,
) -> Optional[bytes]:
    if not url:
        return None
    url_lower = url.lower()
    blocked_ext = (".svg", ".json", ".js")
    if url_lower.endswith(blocked_ext):
        logger.debug("Skip fetch for %s (blocked extension)", url)
        return None
    if not allow_webp_gif and url_lower.endswith((".webp", ".gif")):
        return None
    try:
        resp = http_get(url, referer=referer, timeout=timeout)
    except Exception as exc:
        logger.debug("HTTP fetch failed for %s: %s", url, exc)
        return None
    content_type = (resp.headers.get("Content-Type") or "").lower()
    disallowed_tokens = ["svg", "javascript", "json"]
    if not allow_webp_gif:
        disallowed_tokens.extend(["gif", "webp"])
    if any(token in content_type for token in disallowed_tokens):
        return None
    data = resp.content
    if not data or len(data) > MAX_IMAGE_BYTES:
        return None
    return data


def phash_from_bytes(data: bytes) -> Optional[str]:
    if not data:
        return None
    try:
        with Image.open(io.BytesIO(data)) as image:
            return str(imagehash.phash(image.convert("RGB")))
    except Exception as exc:
        logger.debug("pHash (bytes) failed: %s", exc)
        return None


def phash_from_file(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    try:
        with Image.open(path) as image:
            return str(imagehash.phash(image.convert("RGB")))
    except Exception as exc:
        logger.debug("pHash (file) failed for %s: %s", path, exc)
        return None


def phash_distance(a: Optional[str], b: Optional[str]) -> Optional[int]:
    if not a or not b:
        return None
    try:
        return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)
    except Exception as exc:
        logger.debug("pHash distance error: %s", exc)
        return None


def classify_ad(
    site: str,
    src: str,
    landing_url: str,
    width: int,
    height: int,
    context: str = "",
    *,
    min_score: int = 3,
) -> Tuple[str, int, str]:
    src_lower = (src or "").lower()
    context_lower = (context or "").lower()
    landing_lower = (landing_url or "").lower()

    for token in BLOCKED_TOKENS:
        if token and (token in src_lower or token in context_lower or f"/{token}/" in src_lower):
            return "0", 0, f"blocked:{token}"

    score = 0
    reasons: List[str] = []

    if width >= 180 and height >= 100:
        score += 1
        reasons.append("size>=180x100")

    for std_w, std_h in STANDARD_SIZES:
        if abs(width - std_w) <= 20 and abs(height - std_h) <= 20:
            score += 1
            reasons.append(f"near-{std_w}x{std_h}")
            break

    area = width * height
    if area >= 80_000:
        score += 1
        reasons.append("sufficient-area")

    site_host = normalize_site_host(site)
    landing_host = host_from_url(landing_url)
    if landing_host and landing_host != site_host:
        score += 1
        reasons.append("external-landing")

    for host in (host_from_url(src), landing_host):
        if not host:
            continue
        for known in KNOWN_AD_HOSTS:
            if known in host:
                score += 1
                reasons.append(f"ad-host:{known}")
                break

    for token in AD_KEYWORDS:
        if token and token in src_lower:
            score += 1
            reasons.append(f"src-keyword:{token}")
            break
    for token in AD_KEYWORDS:
        if token and token in landing_lower:
            score += 1
            reasons.append(f"landing-keyword:{token}")
            break
    for token in AD_KEYWORDS:
        if token and token in context_lower:
            score += 1
            reasons.append(f"context:{token}")
            break

    status = "1" if score >= min_score else "0"
    return status, score, " | ".join(reasons)


def final_redirect_url(url: str, *, referer: Optional[str] = None, timeout: int = 15) -> str:
    if not url:
        return ""
    try:
        resp = http_get(url, referer=referer, timeout=timeout)
        return resp.url
    except Exception:
        return url


@dataclass
class BannerRecord:
    site: str
    src: str
    landing_url: str
    landing_final_url: str
    width: int
    height: int
    screenshot_path: str
    phash_hex: Optional[str]
    score: int
    reason: str
    is_ad: str
    ad_id: str
    context: str = ""

    @classmethod
    def from_capture(cls, capture: Dict[str, Any], *, min_score: int = 3) -> Optional["BannerRecord"]:
        site = (capture.get("site") or "").strip()
        src = (capture.get("src") or "").strip()
        if not site or not src:
            return None
        landing_url = (capture.get("landing_url") or "").strip()
        landing_final = (capture.get("landing_final_url") or landing_url).strip()
        width = int(capture.get("width") or 0)
        height = int(capture.get("height") or 0)
        screenshot_path = capture.get("screenshot_path") or ""
        context = str(capture.get("context") or "")

        phash_hex = capture.get("phash")
        if not phash_hex:
            img_bytes = capture.get("img_bytes")
            if img_bytes:
                phash_hex = phash_from_bytes(img_bytes)
        if not phash_hex and screenshot_path:
            phash_hex = phash_from_file(screenshot_path)

        is_ad, score, reason = classify_ad(
            site,
            src,
            landing_final or landing_url,
            width,
            height,
            context=context,
            min_score=min_score,
        )

        key = f"{site}|{phash_hex or canonical_src(src)}"
        ad_id = hashlib.sha1(key.encode("utf-8", "ignore")).hexdigest()[:16]

        return cls(
            site=site,
            src=src,
            landing_url=landing_url,
            landing_final_url=landing_final,
            width=width,
            height=height,
            screenshot_path=screenshot_path,
            phash_hex=phash_hex,
            score=score,
            reason=reason,
            is_ad=is_ad,
            ad_id=ad_id,
            context=context,
        )

    def as_ingest_payload(self, screenshot_url: Optional[str]) -> Dict[str, Any]:
        return {
            "ad_id": self.ad_id,
            "site": self.site,
            "status": "active",
            "url": self.landing_final_url or self.landing_url,
            "src_open": self.src,
            "screenshot": screenshot_url or "",
        }


class BannerDeduper:
    def __init__(self, distance_threshold: int = 5) -> None:
        self.distance_threshold = distance_threshold
        self._records: List[BannerRecord] = []
        self._seen_src: set[str] = set()

    def __len__(self) -> int:
        return len(self._records)

    def add(self, record: BannerRecord) -> bool:
        for existing in self._records:
            dist = phash_distance(existing.phash_hex, record.phash_hex)
            if dist is not None and dist <= self.distance_threshold:
                logger.debug(
                    "Duplicate banner by pHash: %s ~ %s (d=%s)",
                    existing.ad_id,
                    record.ad_id,
                    dist,
                )
                return False
        src_key = canonical_src(record.src)
        if src_key and src_key in self._seen_src:
            logger.debug("Duplicate banner by src: %s", record.src)
            return False
        self._records.append(record)
        if src_key:
            self._seen_src.add(src_key)
        return True

    @property
    def records(self) -> Iterable[BannerRecord]:
        return tuple(self._records)


__all__ = [
    "USER_AGENT",
    "AD_KEYWORDS",
    "BLOCKED_TOKENS",
    "KNOWN_AD_HOSTS",
    "STANDARD_SIZES",
    "http_get",
    "http_get_bytes",
    "phash_from_bytes",
    "phash_from_file",
    "phash_distance",
    "classify_ad",
    "BannerRecord",
    "BannerDeduper",
    "normalize_site_host",
    "host_from_url",
    "canonical_src",
    "final_redirect_url",
]
