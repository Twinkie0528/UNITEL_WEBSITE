# ======================================================
# data_connector.py — Graph API + AI summary + export + date-range support
# ======================================================

from __future__ import annotations
import os, json, traceback, datetime, logging, re
from typing import Any, Dict, Optional
from pathlib import Path
import pandas as pd

from .llm_client import ask_llm
from .prompt_builder import build_prompt
from .fb_fetcher import (
    get_page_insights,
    list_posts,
    list_comments,
    get_instagram_metrics,
    get_ads_insights,
)

# ------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = BASE_DIR / "data" / "downloads"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================
# Helper — query-оос since/until автоматаар илрүүлэх
# ======================================================
def extract_dates_from_query(q: str):
    """
    '2025-10-01-ээс 2025-11-01 хүртэл' гэх мэт query-оос автоматаар хугацаа илрүүлнэ.
    """
    q = q.replace("–", "-").replace("—", "-")
    date_pattern = r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})"
    matches = re.findall(date_pattern, q)
    if not matches:
        return None, None

    def fmt(m):
        return f"{int(m[0]):04d}-{int(m[1]):02d}-{int(m[2]):02d}"

    since = fmt(matches[0])
    until = fmt(matches[-1]) if len(matches) > 1 else None
    return since, until


# ======================================================
# AUTO DATA DETECTION + EXPORT
# ======================================================
def fetch_graph_data(query: str, since: Optional[str] = None, until: Optional[str] = None) -> Dict[str, Any]:
    """
    Автомат байдлаар query-оос хамаарч Graph API дата татна.
    Бүх төрөлд Excel + JSON хэлбэрийн татах линк үүсгэнэ.
    """
    q = query.lower()
    print(f"[DataConnector] Graph fetch for query: {q}")

    # --- Query-оос хугацаа илрүүлэх ---
    if not since and not until:
        s, u = extract_dates_from_query(query)
        since, until = s, u
        if since or until:
            print(f"📅 Date range detected: since={since}, until={until}")

    result: Dict[str, Any] = {"type": "unknown", "data": None, "count": 0}

    try:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # ---------- COMMENTS ----------
        if "comment" in q:
            print("→ Fetching Facebook post comments...")
            posts = list_posts(limit=5, since=since, until=until)
            comments = []
            for p in posts:
                pid = p.get("id")
                if not pid:
                    continue
                try:
                    for c in list_comments(pid, limit=200, since=since, until=until):
                        c["post_id"] = pid
                        comments.append(c)
                except Exception as e:
                    print(f"⚠️ Comment fetch error for {pid}: {e}")

            if not comments:
                return {"status": "empty", "message": "⚠️ Коммент олдсонгүй."}

            json_path = DOWNLOAD_DIR / f"fb_comments_{ts}.json"
            xlsx_path = DOWNLOAD_DIR / f"fb_comments_{ts}.xlsx"
            with open(json_path, "w", encoding="utf-8-sig") as f:
                json.dump(comments, f, ensure_ascii=False, indent=2)
            pd.json_normalize(comments).to_excel(xlsx_path, index=False)

            print(f"✅ Saved {len(comments)} comments → {xlsx_path.name}")
            return {
                "status": "ok",
                "type": "facebook_comments",
                "count": len(comments),
                "json_url": f"/downloads/{json_path.name}",
                "xlsx_url": f"/downloads/{xlsx_path.name}",
                "since": since,
                "until": until,
            }

        # ---------- POSTS ----------
        if any(k in q for k in ["post", "feed", "timeline"]):
            print("→ Fetching Facebook posts...")
            posts = list_posts(since=since, until=until, limit=30)
            if not posts:
                return {"status": "empty", "message": "⚠️ Пост олдсонгүй."}
            xlsx_path = DOWNLOAD_DIR / f"fb_posts_{ts}.xlsx"
            pd.json_normalize(posts).to_excel(xlsx_path, index=False)
            return {
                "status": "ok",
                "type": "facebook_posts",
                "count": len(posts),
                "xlsx_url": f"/downloads/{xlsx_path.name}",
                "since": since,
                "until": until,
            }

        # ---------- INSIGHTS ----------
        if any(k in q for k in ["insight", "reach", "impression", "engagement", "follower"]):
            print("→ Fetching Page insights...")
            insights = get_page_insights(since=since, until=until)
            if not insights:
                return {"status": "empty", "message": "⚠️ Insight дата олдсонгүй."}
            xlsx_path = DOWNLOAD_DIR / f"page_insights_{ts}.xlsx"
            pd.json_normalize(insights).to_excel(xlsx_path, index=False)
            return {
                "status": "ok",
                "type": "page_insights",
                "count": len(insights),
                "xlsx_url": f"/downloads/{xlsx_path.name}",
                "since": since,
                "until": until,
            }

        # ---------- INSTAGRAM ----------
        if any(k in q for k in ["instagram", "ig", "story", "reel"]):
            print("→ Fetching Instagram metrics...")
            ig = get_instagram_metrics(since=since, until=until)
            if not ig:
                return {"status": "empty", "message": "⚠️ Instagram дата хоосон байна."}
            xlsx_path = DOWNLOAD_DIR / f"instagram_{ts}.xlsx"
            pd.json_normalize(ig).to_excel(xlsx_path, index=False)
            return {
                "status": "ok",
                "type": "instagram_insights",
                "count": len(ig),
                "xlsx_url": f"/downloads/{xlsx_path.name}",
                "since": since,
                "until": until,
            }

        # ---------- ADS ----------
        if any(k in q for k in ["ad", "spend", "campaign", "click", "conversion", "ctr", "cpc"]):
            print("→ Fetching Ads insights...")
            ads = get_ads_insights(fields="campaign_name,spend,impressions,reach,clicks,actions",
                                   since=since, until=until)
            if not ads or "data" not in ads:
                return {"status": "empty", "message": "⚠️ Ads дата олдсонгүй."}
            df = pd.DataFrame(ads["data"])
            xlsx_path = DOWNLOAD_DIR / f"ads_insights_{ts}.xlsx"
            df.to_excel(xlsx_path, index=False)
            return {
                "status": "ok",
                "type": "ads_insights",
                "count": len(df),
                "xlsx_url": f"/downloads/{xlsx_path.name}",
                "since": since,
                "until": until,
            }

        # ---------- DEFAULT ----------
        print("→ Default fallback: Fetching posts...")
        posts = list_posts(limit=10, since=since, until=until)
        if posts:
            xlsx_path = DOWNLOAD_DIR / f"default_posts_{ts}.xlsx"
            pd.json_normalize(posts).to_excel(xlsx_path, index=False)
            return {
                "status": "ok",
                "type": "facebook_posts",
                "count": len(posts),
                "xlsx_url": f"/downloads/{xlsx_path.name}",
                "since": since,
                "until": until,
            }

        return {"status": "empty", "message": "⚠️ Аль ч төрөлд дата олдсонгүй."}

    except Exception as e:
        logging.exception("[DataConnector] Fetch error")
        return {"status": "failed", "error": str(e)}


# ======================================================
# SUMMARIZER + AI WRAPPERS
# ======================================================
def summarize_api_data(raw: Any, max_items: int = 30) -> str:
    if not raw:
        return "[no_data]"
    try:
        if isinstance(raw, list):
            subset = raw[:max_items]
            return "\n".join(json.dumps(r, ensure_ascii=False) for r in subset)
        elif isinstance(raw, dict):
            return json.dumps(raw, ensure_ascii=False, indent=2)
        else:
            return str(raw)
    except Exception as e:
        return f"[summarize_error:{e}]"

def get_data_for_ai(query: str) -> str:
    data = fetch_graph_data(query)
    return summarize_api_data(data.get("data", []))

def analyze_marketing_query(user_msg: str) -> str:
    data = fetch_graph_data(user_msg)
    mode = data.get("type", "unknown")
    compact_text = summarize_api_data(data.get("data", []))[:100000]
    prompt = build_prompt(
        f"Generate a concise marketing report (type={mode})\n{user_msg}",
        [{"text": compact_text, "meta": f"[{mode}]"}],
        system_hint="[Marketing Data AI Report]",
    )
    return ask_llm(prompt)
