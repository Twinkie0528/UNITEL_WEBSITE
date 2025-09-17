from __future__ import annotations

import hashlib
import io
import logging
import os
from dataclasses import dataclass
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
BLOCKED_TOKENS = ("thumbnail", "thumb", "avatar", "profile", "logo", "icon")
AD_KEYWORDS = ("ad", "ads", "promo", "banner", "sponsor", "zar", "зар", "сурталчилгаа")
KNOWN_AD_HOSTS = (
    "googlesyndication.com",
    "doubleclick.net",
    "adservice.google.com",
    "taboola.com",
    "taboolanews.com",
    "mgid.com",
    "boost.mn",
    "edge.boost.mn",
    "exchange.boost.mn",
    "criteo.com",
    "criteo.net",
    "adobe.com",
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
    value = site.strip()
    if "//" not in value:
        value = f"https://{value}"
    try:
        host = urlparse(value).netloc.lower()
        return host
    except Exception:
        return value.lower()


def host_from_url(url: str) -> str:
    if not url:
        return ""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def http_get_bytes(url: str, *, timeout: int = 15, referer: Optional[str] = None) -> Optional[bytes]:
    if not url:
        return None
    lower = url.lower()
    if lower.endswith((".gif", ".webp")):
        logger.debug("skip fetch for %s (blocked extension)", url)
        return None
    headers = {"User-Agent": USER_AGENT}
    if referer:
        headers["Referer"] = referer
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        content_type = (resp.headers.get("Content-Type") or "").lower()
        if "gif" in content_type or "webp" in content_type:
            return None
        data = resp.content
        if not data or len(data) > MAX_IMAGE_BYTES:
            return None
        return data
    except requests.RequestException as exc:
        logger.debug("image fetch failed for %s: %s", url, exc)
        return None


def phash_from_bytes(data: bytes) -> Optional[str]:
    if not data:
        return None
    try:
        with Image.open(io.BytesIO(data)) as img:
            return str(imagehash.phash(img.convert("RGB")))
    except Exception as exc:
        logger.debug("phash bytes failed: %s", exc)
        return None


def phash_from_file(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    try:
        with Image.open(path) as img:
            return str(imagehash.phash(img.convert("RGB")))
    except Exception as exc:
        logger.debug("phash file failed for %s: %s", path, exc)
        return None


def phash_distance(a: Optional[str], b: Optional[str]) -> Optional[int]:
    if not a or not b:
        return None
    try:
        return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)
    except Exception as exc:
        logger.debug("phash distance error: %s", exc)
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
    score = 0
    reasons: List[str] = []
    src_lower = (src or "").lower()
    context_lower = (context or "").lower()
    landing_lower = (landing_url or "").lower()

    for token in BLOCKED_TOKENS:
        if token and (token in src_lower or token in context_lower or f"/{token}/" in src_lower):
            return ("0", 0, f"blocked:{token}")

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
    src_host = host_from_url(src)
    landing_host = host_from_url(landing_url)

    if landing_host and landing_host != site_host:
        score += 1
        reasons.append("external-landing")

    for host in (src_host, landing_host):
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
        context = capture.get("context") or capture.get("notes") or ""

        phash_hex = capture.get("phash")
        if not phash_hex:
            img_bytes = capture.get("img_bytes")
            phash_hex = phash_from_bytes(img_bytes) if img_bytes else phash_from_file(screenshot_path)

        is_ad, score, reason = classify_ad(
            site,
            src,
            landing_final or landing_url,
            width,
            height,
            context=str(context),
            min_score=min_score,
        )

        key = f"{site}|{phash_hex or src}"
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
            context=str(context),
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

    def __len__(self) -> int:
        return len(self._records)

    def add(self, record: BannerRecord) -> bool:
        for existing in self._records:
            if existing.site != record.site:
                continue
            dist = phash_distance(existing.phash_hex, record.phash_hex)
            if dist is not None and dist <= self.distance_threshold:
                logger.debug(
                    "duplicate banner by phash: %s ~ %s (d=%s)",
                    existing.ad_id,
                    record.ad_id,
                    dist,
                )
                return False
            if existing.src and existing.src == record.src:
                logger.debug("duplicate banner by src: %s", record.src)
                return False
        self._records.append(record)
        return True

    @property
    def records(self) -> Iterable[BannerRecord]:
        return tuple(self._records)


__all__ = [
    "http_get_bytes",
    "phash_from_bytes",
    "phash_from_file",
    "phash_distance",
    "classify_ad",
    "BannerRecord",
    "BannerDeduper",
]
