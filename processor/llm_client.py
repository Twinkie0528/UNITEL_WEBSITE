# ======================================================
# processor/llm_client.py
# HYBRID Context-Aware LLM Client (Final Compatibility Fix)
# (Responses API + ChatCompletion Fallback + History)
# ======================================================

import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# -------------------- CONFIG LOAD --------------------
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MODEL_PRIMARY = os.getenv("OPENAI_MODEL", "gpt-5-mini")
MODEL_FALLBACK = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini") 

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
DISABLE_FALLBACK = os.getenv("OPENAI_DISABLE_FALLBACK", "0") == "1"
MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "4000"))

log = logging.getLogger("llm_client")

# -------------------- INIT CLIENT --------------------
_client = None
if OPENAI_KEY:
    try:
        _client = OpenAI(api_key=OPENAI_KEY)
        log.info(f"✅ OpenAI client initialized (Primary: {MODEL_PRIMARY}, Fallback: {MODEL_FALLBACK})")
    except Exception as e:
        log.error(f"❌ Failed to initialize OpenAI client: {e}")
else:
    log.warning("⚠️ OPENAI_API_KEY not found in .env — LLM responses may fail.")


# ======================================================
# INTERNAL HELPERS (ШИНЭЧЛЭГДСЭН)
# ======================================================

def _call_responses(system_text: str, user_text: str) -> str:
    """
    (1) Үндсэн API (Stateless - түүхгүй)
    """
    if not _client:
        raise OpenAIError("OpenAI түлхүүр тохируулагдаагүй байна.")

    try:
        r = _client.responses.create(
            model=MODEL_PRIMARY,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_text}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
            ],
            max_output_tokens=MAX_TOKENS,
        )
        return (r.output_text or "").strip()
    except Exception as e:
        raise OpenAIError(f"Responses API error: {e}")


def _call_chat(system_text: str, user_text: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    (2) Нөөц API (Stateful - түүхтэй)
    """
    if not _client:
        raise OpenAIError("OpenAI түлхүүр тохируулагдаагүй байна.")

    # (🟡 ТААРУУЛАХ ХЭСЭГ - ТАНЫ ХҮСЭЛТИЙН ДАГУУ НЭМЭГДЛЭЭ)
    # processor.py-с list[str] ирвэл list[dict] болгож хөрвүүлнэ.
    if history and all(isinstance(h, str) for h in history):
        log.warning("History format mismatch (list[str]). Converting to list[dict] (user roles only).")
        # Зөвхөн хэрэглэгчийн асуултууд тул бүгдийг "user" role-той болгоно
        history = [{"role": "user", "content": h} for h in history]
    # (🟡 ТААРУУЛАХ ХЭСЭГ - ТӨГСӨВ)

    messages = []
    
    # 1. System заавар
    messages.append({"role": "system", "content": system_text})

    # 2. History (Ярианы түүх - одоо зөв dict форматаар орно)
    if history:
        for h in history:
            if isinstance(h, dict) and "role" in h and "content" in h:
                messages.append(h)
    
    # 3. Одоогийн асуулт
    messages.append({"role": "user", "content": user_text})

    try:
        r = _client.chat.completions.create(
            model=MODEL_FALLBACK, 
            messages=messages,
            temperature=0.5,
            max_tokens=MAX_TOKENS,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        raise OpenAIError(f"ChatCompletion fallback error: {e}")


# ======================================================
# MAIN WRAPPER
# ======================================================
def ask_llm(prompt: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Hybrid Context-Aware LLM wrapper:
    - Хэрэв 'history' байвал, шууд stateful _call_chat-г дуудна.
    - Хэрэв 'history' байхгүй бол, stateless _call_responses-г оролдож, 
      алдаа гарвал _call_chat руу fallback хийнэ.
    """
    if not _client:
        return "⚠️ OpenAI түлхүүр тохируулагдаагүй байна. .env дахь OPENAI_API_KEY-г шалгана уу."

    system_text = (
        "Та бол Unitel AI Assistant — хэрэглэгчийн оруулсан өгөгдөл, файл, ярианы түүхийг үндэслэн "
        "Монгол болон Англи хэлээр ойлгомжтой, Markdown форматтайгаар хариулт өгдөг мэргэжлийн туслах юм."
    )

    try:
        # --- (A) CONTEXT-AWARE (Түүхтэй) ---
        if history:
            log.info("Context detected. Using stateful chat completion API (_call_chat).")
            # _call_chat нь дотроо history-г хөрвүүлэх логиктой болсон
            return _call_chat(system_text, prompt, history=history)

        # --- (B) STATELESS (Түүхгүй, шинэ асуулт) ---
        log.info("No context. Using stateless hybrid logic (_call_responses -> _call_chat).")
        try:
            # 1. Үндсэн API-г оролдох (Stateless)
            return _call_responses(system_text, prompt)
        
        except OpenAIError as e:
            msg = str(e)
            log.warning(f"⚠️ LLM primary error: {msg}")

            bad_request = any(x in msg for x in ["400", "invalid_request_error", "Unsupported", "Responses API error"])
            
            if not DISABLE_FALLBACK and bad_request:
                log.info("Fallback activated. Calling _call_chat (stateless).")
                # Нөөц API-г дуудах (history-гүйгээр)
                return _call_chat(system_text, prompt, history=None)
            
            raise e

    # --- (C) ЕРӨНХИЙ АЛДААНЫ УДИРДЛАГА ---
    except OpenAIError as e:
        msg = str(e)
        if "insufficient_quota" in msg:
            return "⚠️ OpenAI квот дууссан байна."
        if "rate_limit" in msg:
            return "⚠️ Хэт олон хүсэлт илгээгдэж байна. Түр хүлээгээд дахин оролдоно уу."
        log.error(f"❌ LLM Error: {msg}")
        return f"⚠️ LLM алдаа: {msg}"

    except Exception as e:
        log.exception("❌ Unexpected LLM system exception: %s", e)
        return f"⚠️ LLM системийн алдаа: {e}"


# ======================================================
# QUICK LOCAL TEST
# ======================================================
if __name__ == "__main__":
    print("🔍 LLM Hybrid Context-Aware Test:")
    
    # Test 1: Stateless (No History)
    print("\n--- Test 1: Stateless (No History) ---")
    print(f"ASSISTANT: {ask_llm('Сайн уу? Энэ систем ажиллаж байна уу?')}")

    # Test 2: Stateful (With dict History - зөв ажиллах ёстой)
    print("\n--- Test 2: Stateful (Correct dict History) ---")
    dict_history = [
        {"role": "user", "content": "Миний хамгийн дуртай өнгө бол цэнхэр."},
        {"role": "assistant", "content": "Ойлголоо, таны дуртай өнгө цэнхэр юм байна."}
    ]
    print(f"ASSISTANT: {ask_llm('Миний дуртай өнгө юу вэ?', history=dict_history)}")

    # Test 3: Stateful (With str History - хөрвүүлэх ёстой)
    print("\n--- Test 3: Stateful (Incorrect str History - Auto-converting) ---")
    str_history = [
        "Миний нэрийг Болд гэдэг."
    ]
    # Энэ асуултад "Болд" гэж хариулах ёстой
    print(f"ASSISTANT: {ask_llm('Миний нэр хэн бэ?', history=str_history)}")