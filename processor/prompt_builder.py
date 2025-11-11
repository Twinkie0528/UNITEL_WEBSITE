# ======================================================
# processor/prompt_builder.py
# Universal Prompt Builder for Unitel AI Assistant
# Supports: text, structured (csv/json/xlsx), API, sentiment, numeric
# Author: Unitel AI Hub (2025 edition)
# ======================================================

from __future__ import annotations
import json
from typing import List, Tuple, Union

# ======================================================
# DEFAULT PROMPT GUIDELINES
# ======================================================

GENERAL_INSTRUCTIONS = """
Та Unitel Assistant. Markdown формат ашиглан цэгцтэй, ойлгомжтой, товч хариулт бич.
Тайлбар, анализ, дүгнэлт бүрийг бодитой, дата дээр тулгуурлан гарга.

## Хариулт дараах бүтэцтэй байна:
### 🧩 Тойм
- 2–3 өгүүлбэрт гол санааг нэгтгэн бич.

### 📊 Гол санаа
- 3–7 bullet байдлаар гол мэдээлэл, дүн, санаануудыг гарга.
- Хэрвээ өгөгдөл тоон шинжтэй бол хүснэгтээр харуулж болно.

### 💡 Дүгнэлт
- 1 догол мөрөнд гол санааг нэгтгэ.
"""

SENTIMENT_INSTRUCTIONS = """
Та хэрэглэгчийн сэтгэгдэл, үнэлгээний чанарын болон тоон шинжилгээг хийнэ.
Markdown форматтайгаар дараах бүтэцтэй бич:

## Нийт хандлагын дүн
- Эерэг: X%
- Саармаг: Y%
- Сөрөг: Z%
**Дүгнэлт:** Хандлагын чиглэл ба шалтгаан.

## Гол сэдвүүд (3–5)
1. **Сэдэв нэр** — гол санаа ба хандлага.
2. **Сэдэв нэр** — …

## Төлөөлөх ишлэлүүд (3–6)
- “...”
- “...”

## Insight ба Зөвлөмж
- Гол ойлголт, сайжруулалт, чиглэл.
"""

NUMERIC_INSTRUCTIONS = """
Та өгөгдлийг тоон талаас нь дүгнэ.
Хариултаа markdown форматтай дараах бүтэцтэй бич:

## Гол үзүүлэлтүүд
| Үзүүлэлт | Утга | Тайлбар |
|-----------|------|----------|
| … | … | … |

## Хандлагын дүгнэлт
- Гол өөрчлөлт, өсөлт/бууралт
- Боломжит хамаарал, шалтгаан

## Зөвлөмж
- 2–3 actionable санал гарга.
"""

STRUCTURED_DATA_INSTRUCTIONS = """
Та дараах structured өгөгдлийг шинжилж, дүгнэлт гарга.
Файлын бүтэц, талбарууд, хандлага, хамаарал, гол мэдээлэл, 
мөн боломжит утга, статистикийг дүгнэ.
"""

# ======================================================
# HELPER
# ======================================================

def truncate_text(text: str, limit: int = 120_000) -> str:
    """LLM рүү илгээх текстийг багасгаж таслах."""
    if not text:
        return ""
    return text[:limit] + ("\n...[TRUNCATED]..." if len(text) > limit else "")

def safe_json(obj: Union[dict, list]) -> str:
    """JSON-г текст хэлбэрт аюулгүй хөрвүүлэх."""
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)

# ======================================================
# PROMPT BUILDERS
# ======================================================

def build_general_prompt(user_msg: str, materials: str | None = None) -> str:
    """Ерөнхий сэдвийн prompt."""
    parts = [GENERAL_INSTRUCTIONS.strip()]
    if user_msg:
        parts.append(f"## 🧠 Асуулт\n{user_msg.strip()}")
    if materials:
        parts.append(f"## 📁 Материал\n{truncate_text(materials)}")
    return "\n\n".join(parts)

def build_sentiment_prompt(user_msg: str, data_blobs: List[Tuple[str, str]]) -> str:
    """Сэтгэгдлийн анализ prompt."""
    parts = [SENTIMENT_INSTRUCTIONS.strip()]
    if user_msg:
        parts.append(f"## Хэрэглэгчийн асуулт\n{user_msg.strip()}")
    for text, meta in data_blobs:
        parts.append(f"### Файл: {meta}\n{truncate_text(text, 100_000)}")
    return "\n\n".join(parts)

def build_numeric_prompt(user_msg: str, json_data: dict | list) -> str:
    """Тоон дүн шинжилгээ хийхэд зориулсан prompt."""
    json_str = safe_json(json_data)
    parts = [NUMERIC_INSTRUCTIONS.strip()]
    parts.append(f"## Асуулт\n{user_msg.strip()}")
    parts.append(f"## Өгөгдөл\n{truncate_text(json_str, 80_000)}")
    return "\n\n".join(parts)

def build_structured_prompt(user_msg: str, records: list[dict], meta: str = "") -> str:
    """Structured өгөгдөл (жишээ нь CSV, Excel, JSON)-д зориулсан prompt."""
    
    # ✅ ====================================================
    # ✅ САЙЖРУУЛАЛТ 2 (Таны саналаар + Нэмэлт)
    # ✅ ====================================================
    total_rows = len(records)
    # 10 хүртэлх мөрийг дээж болгон харуулна
    sample_size = min(total_rows, 10) 
    preview = safe_json(records[:sample_size]) if records else "(хоосон хүснэгт)"
    
    # 💡 Нэмэлт сайжруулалт: "Нийт хэдэн мөр байна?" гэдэгт хариулахын тулд
    # нийт мөрийн тоог мета-д автоматаар нэмэв.
    full_meta = f"{meta} (Нийт: {total_rows} мөр)"
    
    return f"{STRUCTURED_DATA_INSTRUCTIONS}\n\n## Асуулт\n{user_msg}\n\n## Мета\n{full_meta}\n\n## Өгөгдлийн дээж (Эхний {sample_size} мөр)\n{truncate_text(preview, 50_000)}"


from typing import Union

# ✅ ====================================================
# ✅ САЙЖРУУЛАЛТ 1 (Өмнөх хүсэлт)
# ✅ ====================================================
def has_textual_field(data: Union[str, list, dict]) -> bool:
    """Data дотор text/comment-like талбар (эсвэл өөрөө текст) байгаа эсэхийг шалгана."""
    
    # Case 1: Data is a list of records (from Excel/CSV/JSON array)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        sample_keys = " ".join(data[0].keys()).lower()
        # Check for common text field names
        if any(k in sample_keys for k in ["comment", "review", "feedback", "text", "caption", "body", "message", "сэтгэгдэл", "тайлбар"]):
            return True
    
    # Case 2: Data is a single block of text (from .txt or a single cell)
    if isinstance(data, str):
        return len(data) > 20 # Assume any non-trivial string is "textual"

    return False


# ---------- INTENT DETECTOR ----------
def detect_intent(user_msg: str, meta: str = "", records: list | None = None) -> str:
    """
    Файл болон хэрэглэгчийн асуултаас intent тодорхойлох.
    """
    msg = (user_msg or "").lower()
    name = (meta or "").lower()

    # 1️⃣ Sentiment / Comment
    if any(k in msg for k in ["sentiment", "сэтгэгдэл", "reaction", "feedback", "comment", "tone", "эерэг", "сөрөг"]):
        return "sentiment"

    # 2️⃣ Influencer / Marketing dataset
    if any(k in name for k in ["influencer", "impression", "reach", "view", "performance"]) or \
       any(k in msg for k in ["influencer", "импрешн", "reach", "view", "брэнд", "influence"]):
        return "influencer"

    # 3️⃣ Advertising dataset
    if any(k in name for k in ["ads", "banner", "ad_report", "campaign"]) or "ads" in msg:
        return "ad_report"

    # 4️⃣ Numeric / Statistical request
    if any(k in msg for k in ["stat", "тоо", "growth", "spend", "click", "performance", "rate", "data"]):
        return "numeric"

    # 5️⃣ Structured data check (list of dict)
    if records and all(isinstance(x, dict) for x in records):
        return "influencer"

    # Default
    return "general"



# ---------- BUILD PROMPT ----------
def build_prompt(user_msg: str, data: Union[str, list, dict], meta: str = "") -> str:
    """
    Автомат prompt сонгох:
    - detect_intent() ашиглан file/content/context-аас intent тодорхойлох
    - “influencer”, “ads”, “sentiment”, “numeric”, “general” төрлөөр төрөлжүүлнэ
    """
    msg = user_msg.lower().strip()

    # 1️⃣ Intent тодорхойлох
    intent = detect_intent(user_msg, meta, data if isinstance(data, list) else None)

    # ✅ ====================================================
    # ✅ ЛОГИКИЙН САЙЖРУУЛАЛТ ХИЙГДСЭН ХЭСЭГ (САЙЖРУУЛАЛТ 1)
    # ✅ ====================================================
    if intent == "sentiment":
        # ШАЛГАЛТ: Sentiment intent-тэй ч, өгөгдөл нь текстэн талбартай эсэх
        if has_textual_field(data):
            # Текст талбар байна -> Sentiment анализ хий
            blob = [(data if isinstance(data, str) else safe_json(data)), meta]
            return build_sentiment_prompt(user_msg, [blob])
        else:
            # Текст талбар байхгүй (Жишээ нь: Influencer_data.xlsx)
            # -> Sentiment биш, STRUCTURED эсвэл NUMERIC prompt руу шилжүүл
            if isinstance(data, list) and all(isinstance(x, dict) for x in data):
                 # Энэ нь Influencer_data.xlsx шиг файлуудыг зөв барьж авна
                return build_structured_prompt(user_msg, data, meta)
            else:
                # Бусад тоон өгөгдөл
                return build_numeric_prompt(user_msg, data)
    # ✅ ====================================================
    # ✅ САЙЖРУУЛАЛТ ДУУСАВ
    # ✅ ====================================================

    elif intent == "influencer":
        # Influencer / Structured data
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            return build_structured_prompt(user_msg, data, meta)
        else:
            materials = data if isinstance(data, str) else safe_json(data)
            return build_general_prompt(user_msg, materials) # Fallback to general if not a list of dicts

    elif intent == "ad_report":
        # Ad performance dataset
        if isinstance(data, (list, dict)):
            return build_numeric_prompt(user_msg, data)
        else:
            return build_general_prompt(user_msg, str(data)) # Fallback

    elif intent == "numeric":
        if isinstance(data, (list, dict)):
            return build_numeric_prompt(user_msg, data)
        else:
            return build_general_prompt(user_msg, str(data)) # Fallback

    else:
        # Default general prompt
        materials = data if isinstance(data, str) else safe_json(data)
        return build_general_prompt(user_msg, materials)


# ======================================================
# QUICK TEST
# ======================================================
if __name__ == "__main__":
    # 1. Сэтгэгдэлтэй өгөгдөл (Зөв ажиллах ёстой)
    test_msg_1 = "Сэтгэгдлийн ерөнхий хандлагыг дүгнэ."
    test_data_1 = [
        {"comment": "Unitel-ийн үйлчилгээ сайжирсан байна!", "sentiment": "positive"},
        {"comment": "Data хурдан дуусч байна.", "sentiment": "negative"}
    ]
    print("--- TEST 1 (Sentiment) ---")
    prompt_1 = build_prompt(test_msg_1, test_data_1, "user_feedback.json")
    print(prompt_1)
    assert SENTIMENT_INSTRUCTIONS in prompt_1

    # 2. Сэтгэгдэлгүй, тоон өгөгдөл (Сайжруулсан логик шалгах)
    test_msg_2 = "Эдгээр сэтгэгдлүүдийг дүгнээд өг."
    test_data_2 = [
        {"influencer": "UserA", "followers": 10000, "views": 50000},
        {"influencer": "UserB", "followers": 5000, "views": 10000},
        {"influencer": "UserC", "followers": 1, "views": 1},
        {"influencer": "UserD", "followers": 1, "views": 1},
        {"influencer": "UserE", "followers": 1, "views": 1},
        {"influencer": "UserF", "followers": 1, "views": 1},
        {"influencer": "UserG", "followers": 1, "views": 1},
        {"influencer": "UserH", "followers": 1, "views": 1},
        {"influencer": "UserI", "followers": 1, "views": 1},
        {"influencer": "UserJ", "followers": 1, "views": 1},
        {"influencer": "UserK", "followers": 1, "views": 1},
    ]
    print("\n--- TEST 2 (Influencer Data - Асуудалт кэйс) ---")
    prompt_2 = build_prompt(test_msg_2, test_data_2, "influencer_data.xlsx")
    print(prompt_2)
    
    # Шалгалт: Энэ нь SENTIMENT БИШ, харин STRUCTURED байх ёстой
    assert SENTIMENT_INSTRUCTIONS not in prompt_2
    assert STRUCTURED_DATA_INSTRUCTIONS in prompt_2
    # Шалгалт (Сайжруулалт 2): Нийт мөрийн тоо (11) болон дээж (10) зөв орсон эсэх
    assert "(Нийт: 11 мөр)" in prompt_2
    assert "Өгөгдлийн дээж (Эхний 10 мөр)" in prompt_2
    
    print("\n✅ Тест амжилттай: Sentiment асуусан ч текст талбар байхгүй тул STRUCTURED prompt сонгогдлоо.")
    print("✅ Тест амжилттай: Structured prompt нь нийт мөрийн тоо (11) болон дээж (10)-ийг зөв тусгалаа.")