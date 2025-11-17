# ======================================================
# processor/__init__.py — Unified Chat Processor (Context-Aware)
# (Засварласан, TF-ийн үлдэгдэлгүй хувилбар)
# ======================================================

from __future__ import annotations
import os, json, random, pickle, re, numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# --- Нэгтгэсэн Imports ---
from .file_handler import extract_text_from_file, extract_records_from_file
from .prompt_builder import (
    build_prompt, 
    build_general_prompt,
    build_sentiment_prompt, # (Шинэ хувилбараас)
    # detect_intent,        # (❗ ЗАCВАР: TF-ийн үлдэгдэл устгагдсан)
    has_textual_field,      # (Хуучин хувилбараас)
    safe_json               # (Хуучин хувилбараас)
)
from .llm_client import ask_llm
# --- ӨӨРЧЛӨГДСӨН IMPORT ---
from .sentiment_analyzer import analyze_sentiment, analyze_sentiment_ai
from .data_connector import fetch_graph_data
# GPT intent engine (шинэ)
from .intent_classifier import classify_intent, get_intent_response

# -------- CONTEXT MEMORY (2.1) -----------
# (Шинэ хувилбараас)
USER_CONTEXTS = {}

def get_user_context(session_id: str):
    """Сешн бүрийн хэрэглэгчийн context (history, last_file) хадгалах."""
    return USER_CONTEXTS.get(session_id, {"history": [], "last_file": None})

def update_user_context(session_id: str, message: str, file_meta: str = None):
    ctx = USER_CONTEXTS.setdefault(session_id, {"history": [], "last_file": None})
    ctx["history"].append(message) 
    if file_meta:
        ctx["last_file"] = file_meta
    ctx["history"] = ctx["history"][-20:]

# (Хуучин TensorFlow-ийн setup болон utility функцууд энд байхгүй)

# ======================================================
# MAIN CHAT PROCESSOR (НЭГТГЭСЭН)
# ======================================================
def process_query(msg: str,
                  session_id: str, # <--- (Шинэ)
                  files: Optional[List[str]] = None,
                  user: Optional[str] = None) -> str:
    """
    Unified logic for Chatbot: (Context-Aware)
    """
    msg = (msg or "").strip()
    if not msg:
        return "⚠️ Хоосон асуулт байна."

    # --- (2.2a) Context авах (Шинэ хувилбараас) ---
    context = get_user_context(session_id)
    history = context.get("history", [])
    last_file_meta = context.get("last_file")
    # ---------------------------

    files = files or []
    lower = msg.lower()

    # ---------- 1️⃣ FILE ATTACHED (2.2b) ----------
    if files:
        all_records: list[dict] = []
        blobs: list[tuple[str, str]] = []
        file_meta = "Uploaded File"

        # (Хуучин кодын логик: record/blob ялгах)
        for f in files:
            try:
                recs, meta1 = extract_records_from_file(f)
                if recs:
                    all_records.extend(recs)
                    file_meta = meta1
                else:
                    text, meta2 = extract_text_from_file(f)
                    blobs.append((text, meta2))
                    file_meta = meta2
            except Exception as e:
                blobs.append((f"[extract_error:{e}]", f"[{Path(f).name}]"))

        # (Хуучин кодын логик: sentiment тодорхойлох)
        wants_sentiment = any(k in lower for k in [
            "sentiment","сэтгэгдэл","feedback","comment","tone","эерэг","сөрөг"
        ])

        # (Хуучин кодын логик: Хүснэгтэн өгөгдөл боловсруулах)
        if all_records:
            # ❗ ЗАCВАР: intent_guess = detect_intent(...) мөрийг устгасан (TF-ийн үлдэгдэл)
            textish = has_textual_field(all_records)

            # --- СЭТГЭГДЭЛТЭЙ ФАЙЛ: (AI / ЛОКАЛ) ---
            if wants_sentiment and textish:
                # === AI буюу LLM-д суурилсан хувилбар ===
                USE_AI_SENTIMENT = True   # <== toggle: True бол AI ашиглана, False бол хуучин Lexicon

                if USE_AI_SENTIMENT:
                    answer = analyze_sentiment_ai(all_records, meta=file_meta)
                    update_user_context(session_id, msg, file_meta=file_meta)
                    return answer
                else:
                    # (Хуучин локал sentiment-ийн логик)
                    s = analyze_sentiment(all_records)
                    payload = { "meta": file_meta, "counts": s["counts"], "ratios": s["ratios"], "examples": s["examples"] }
                    prompt = (
                        "Доорх нь хэрэглэгчийн СЭТГЭГДЛИЙН бодит тооцоо (локал) юм. "
                        "Тоонуудыг өөрчлөхгүй. Markdown тайлан бичиж, хувь, дүгнэлт, 3–5 гол сэдэв, "
                        "төлөөлөх ишлэлүүдийг оруул. Монгол хэлээр бич.\n\n"
                        + safe_json(payload)
                    )
                    answer = ask_llm(prompt, history=history)
                    update_user_context(session_id, msg, file_meta=file_meta)
                    return answer
            
            # ❗ ЗАCВАР: Problem 4 - 'intent_guess'-д суурилсан 'if' нөхцлийг 'else' болгож өөрчилсөн
            else:
                # --- STRUCTURED буюу тоон өгөгдөл (Сэтгэгдэл биш үед) ---
                # Энэ бол сэтгэгдэл биш, эсвэл сэтгэгдэл ч текстэн талбар байхгүй
                # тохиолдолд ажиллах ердийн structured data-н боловсруулалт.
                prompt = build_prompt(msg, all_records, meta=file_meta)
                answer = ask_llm(prompt, history=history)
                update_user_context(session_id, msg, file_meta=file_meta)
                return answer

        # --- Таблиц биш / текстэн файл ---
        if blobs:
            prompt = build_prompt(msg, [ {"blob": t, "meta": m} for (t,m) in blobs ], meta=file_meta)
            answer = ask_llm(prompt, history=history)
            update_user_context(session_id, msg, file_meta=file_meta)
            return answer

        # --- Fallback (Файл боловсруулж чадаагүй) ---
        prompt = build_general_prompt(msg, f"[{file_meta}] structured/бус материал танигдсангүй.")
        answer = ask_llm(prompt, history=history)
        update_user_context(session_id, msg, file_meta=file_meta)
        return answer

    # ---------- 2️⃣ FACEBOOK / INSIGHT QUERIES ----------
    fb_keywords = ["facebook", "insight", "impression", "reach", "spend", "campaign", "ad", "graph"]
    comment_keywords = ["facebook comment", "fb comment", "коммент тат", "comment тат", "коммент өг"]

    if any(k in lower for k in fb_keywords + comment_keywords):
        try:
            result = fetch_graph_data(msg)
            if not result:
                return "⚠️ Graph API-аас өгөгдөл буцаагдсангүй."

            links = []
            if result.get("xlsx_url"):
                links.append(f'📘 <a href="{result["xlsx_url"]}" download target="_blank">XLSX татах</a>')
            if result.get("json_url"):
                links.append(f'🧾 <a href="{result["json_url"]}" download target="_blank">JSON татах</a>')
            links_html = "<br>".join(links)

            if "comment" in lower:
                resp = f"✅ {result.get('count', 0)} коммент татлаа.<br>{links_html}"
            elif "ad" in lower or "insight" in lower:
                resp = f"📊 Ads тайлан гаргалаа ({result.get('count', 0)} мөр).<br>{links_html}"
            else:
                resp = f"✅ {result.get('count', 0)} бичлэг татлаа.<br>{links_html}"
            
            return resp

        except Exception as e:
            return f"⚠️ Facebook Graph API дата татахад алдаа: {e}"

    # ❗ ЗАCВАР: Problem 2 - Энд байсан '# 3 SENTIMENT DIRECT' блокыг устгасан.
    # (Учир нь 'analyze_sentiment' зөвхөн файл дээр ажиллах ёстой)

    # ---------- 3️⃣ GPT INTENT (Хуучин 4-р блок) ----------
    tag = classify_intent(msg)
    if tag and tag != "none":
        ans = get_intent_response(tag)
        if ans:
            update_user_context(session_id, msg, file_meta=last_file_meta)
            return ans

    # ---------- 4️⃣ FALLBACK → LLM (Хуучин 5-р блок) ----------
    # (2.2c) Файлгүй үед last_file_meta-г ашиглах
    blobs = []
    if last_file_meta:
        blobs.append((f"[Тайлбар: Хэрэглэгч өмнө нь ашигласан '{last_file_meta}' файлын талаар асууж байна]", last_file_meta))

    base_prompt = build_general_prompt(msg, blobs)

    # (2.2c) History-г ашиглах (❗ ЗАCВАР: Problem 3 - 'history'-г зөвхөн prompt-д нэгтгэсэн)
    if history:
        ctx = "\n".join(history[-8:]) 
        final_prompt = f"Өмнөх ярианы товч түүх:\n{ctx}\n\nШинэ асуулт:\n{base_prompt}"
    else:
        final_prompt = base_prompt

    # 'history'-г prompt-д оруулсан тул 'ask_llm'-д 'history' параметрийг дамжуулахгүй
    answer = ask_llm(final_prompt) 

    # --- (2.2c) Context шинэчлэх ---
    update_user_context(session_id, msg, file_meta=last_file_meta) 
    
    return answer