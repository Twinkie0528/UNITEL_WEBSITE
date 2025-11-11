# ======================================================
# processor/__init__.py — Unified Chat Processor (Context-Aware)
# (Хоёр хувилбарыг нэгтгэсэн)
# ======================================================

from __future__ import annotations
import os, json, random, pickle, re, numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize

# --- Нэгтгэсэн Imports ---
from .file_handler import extract_text_from_file, extract_records_from_file
from .prompt_builder import (
    build_prompt, 
    build_general_prompt,
    build_sentiment_prompt, # (Шинэ хувилбараас)
    detect_intent,          # (Хуучин хувилбараас)
    has_textual_field,      # (Хуучин хувилбараас)
    safe_json               # (Хуучин хувилбараас)
)
from .llm_client import ask_llm
# --- ӨӨРЧЛӨГДСӨН IMPORT ---
from .sentiment_analyzer import analyze_sentiment, analyze_sentiment_ai
from .data_connector import fetch_graph_data

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

# ======================================================
# INITIAL SETUP
# (Шинэ хувилбараас)
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent
lemmatizer = WordNetLemmatizer()

try:
    intents = json.loads((BASE_DIR / "job_intents.json").read_text(encoding="utf-8"))
    words = pickle.load(open(BASE_DIR / "words.pkl", "rb"))
    classes = pickle.load(open(BASE_DIR / "classes.pkl", "rb"))
    model = load_model(BASE_DIR / "chatbot_model.h5")
except Exception:
    intents = words = classes = model = None


# ======================================================
# INTENT UTILITIES
# (Шинэ хувилбараас)
# ======================================================
def clean_sentence(s: str) -> list[str]:
    toks = wordpunct_tokenize(s)
    return [lemmatizer.lemmatize(w.lower()) for w in toks]

def bow(sentence: str, vocab: list[str]):
    sw = clean_sentence(sentence)
    return [1 if w in sw else 0 for w in vocab]

def predict_intent(sentence: str, threshold=0.4):
    if not model or not words or not classes:
        return []
    res = model.predict(np.array([bow(sentence, words)]), verbose=0)[0]
    ranked = [[i, r] for i, r in enumerate(res) if r > threshold]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[i], "p": float(p)} for i, p in ranked]

def intent_reply(msg: str) -> Optional[str]:
    ranked = predict_intent(msg, threshold=float(os.getenv("UNITEL_INTENT_THRESHOLD", "0.72")))
    if not ranked:
        return None
    top = ranked[0]["intent"]
    for it in intents.get("intents", []):
        if it.get("tag") == top and it.get("responses"):
            return random.choice(it["responses"])
    return None


# ======================================================
# MAIN CHAT PROCESSOR (НЭГТГЭСЭН)
# ======================================================
def process_query(msg: str,
                  session_id: str, # <--- (Шинэ)
                  files: Optional[List[str]] = None,
                  user: Optional[str] = None) -> str:
    """
    Unified logic for Chatbot: (Context-Aware)
    - (2.2a) session_id ашиглан context (history, last_file) авна
    - (2.2b) Файлтай үед context-г шинэчилнэ (last_file) - (Хуучин кодын нарийвчилсан логик ашиглана)
    - (2.2c) Файлгүй үед context-с (last_file, history) ашиглана
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
    # (Хуучин хувилбарын нарийвчилсан файл боловсруулах логикийг
    # Шинэ хувилбарын context/history-тэй нэгтгэв)
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
            intent_guess = detect_intent(msg, file_meta, all_records)
            textish = has_textual_field(all_records)

            # --- СЭТГЭГДЭЛТЭЙ ФАЙЛ: (AI / ЛОКАЛ) ---
            # (ЭНЭ ХЭСЭГ ТАНЫ ХҮСЭЛТИЙН ДАГУУ ӨӨРЧЛӨГДЛӨӨ)
            if wants_sentiment and textish:
                # === AI буюу LLM-д суурилсан хувилбар ===
                USE_AI_SENTIMENT = True   # <== toggle: True бол AI ашиглана, False бол хуучин Lexicon

                if USE_AI_SENTIMENT:
                    answer = analyze_sentiment_ai(all_records, meta=file_meta)
                    update_user_context(session_id, msg, file_meta=file_meta)
                    return answer
                else:
                    s = analyze_sentiment(all_records)
                    payload = {
                        "meta": file_meta,
                        "counts": s["counts"],
                        "ratios": s["ratios"],
                        "examples": s["examples"],
                    }
                    prompt = (
                        "Доорх нь хэрэглэгчийн СЭТГЭГДЛИЙН бодит тооцоо (локал) юм. "
                        "Тоонуудыг өөрчлөхгүй. Markdown тайлан бичиж, хувь, дүгнэлт, 3–5 гол сэдэв, "
                        "төлөөлөх ишлэлүүдийг оруул. Монгол хэлээр бич.\n\n"
                        + safe_json(payload)
                    )
                    answer = ask_llm(prompt, history=history)
                    update_user_context(session_id, msg, file_meta=file_meta)
                    return answer

            # --- STRUCTURED буюу тоон өгөгдөл ---
            if intent_guess in ("influencer", "ad_report", "numeric", "general"):
                prompt = build_prompt(msg, all_records, meta=file_meta)
                # (Нэгтгэсэн: history болон session_id ашиглах)
                answer = ask_llm(prompt, history=history)
                update_user_context(session_id, msg, file_meta=file_meta)
                return answer

        # --- Таблич биш / текстэн файл ---
        # (Хуучин кодын логик: Blob боловсруулах)
        if blobs:
            prompt = build_prompt(msg, [ {"blob": t, "meta": m} for (t,m) in blobs ], meta=file_meta)
            # (Нэгтгэсэн: history болон session_id ашиглах)
            answer = ask_llm(prompt, history=history)
            update_user_context(session_id, msg, file_meta=file_meta)
            return answer

        # --- Fallback (Файл боловсруулж чадаагүй) ---
        prompt = build_general_prompt(msg, f"[{file_meta}] structured/бус материал танигдсангүй.")
        # (Нэгтгэсэн: history болон session_id ашиглах)
        answer = ask_llm(prompt, history=history)
        update_user_context(session_id, msg, file_meta=file_meta)
        return answer

    # ---------- 2️⃣ FACEBOOK / INSIGHT QUERIES ----------
    # (Шинэ хувилбараас, context ашиглахгүй хэсэг)
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
            
            # Энэ нь LLM биш тул context-д хадгалахгүй (эсвэл асуултыг хадгалж болно)
            # update_user_context(session_id, msg) 
            return resp

        except Exception as e:
            return f"⚠️ Facebook Graph API дата татахад алдаа: {e}"

    # ---------- 3️⃣ SENTIMENT DIRECT ----------
    # (Шинэ хувилбараас, context-д хадгална)
    if re.search(r"(sentiment|сэтгэгдэл|positive|negative|эерэг|сөрөг)", msg, re.I):
        ans = analyze_sentiment(msg)
        update_user_context(session_id, msg, file_meta=last_file_meta)
        return ans

    # ---------- 4️⃣ LOCAL INTENT ----------
    # (Шинэ хувилбараас, context-д хадгална)
    ans = intent_reply(msg)
    if ans:
        update_user_context(session_id, msg, file_meta=last_file_meta)
        return ans

    # ---------- 5️⃣ FALLBACK → LLM (2.2c) ----------
    # (Шинэ хувилбараас - context ашиглан хариулах)
    
    # (2.2c) Файлгүй үед last_file_meta-г ашиглах
    blobs = []
    if last_file_meta:
        blobs.append((f"[Тайлбар: Хэрэглэгч өмнө нь ашигласан '{last_file_meta}' файлын талаар асууж байна]", last_file_meta))

    base_prompt = build_general_prompt(msg, blobs)

    # (2.2c) History-г ашиглах
    if history:
        ctx = "\n".join(history[-8:]) 
        final_prompt = f"Өмнөх ярианы товч түүх:\n{ctx}\n\nШинэ асуулт:\n{base_prompt}"
    else:
        final_prompt = base_prompt

    answer = ask_llm(final_prompt) # History-г prompt-д оруулсан тул энд дамжуулахгүй

    # --- (2.2c) Context шинэчлэх ---
    update_user_context(session_id, msg, file_meta=last_file_meta) 
    # (Хариултыг мөн history-д нэмбэл дараагийн context-д ашиглагдана)
    # update_user_context(session_id, answer, file_meta=last_file_meta) 
    
    return answer