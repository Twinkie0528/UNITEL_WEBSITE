# ======================================================
# processor/sentiment_analyzer.py
# Universal Sentiment Analyzer (Merged: Lexicon + LLM)
# Author: Unitel AI Hub (2025 edition)
# ======================================================

from __future__ import annotations
import re, statistics, json, logging
from typing import List, Dict, Optional, Tuple, Union

from .llm_client import ask_llm
from .prompt_builder import build_sentiment_prompt, build_general_prompt

# Optional imports
try:
    import pandas as pd
except ImportError:
    pd = None

log = logging.getLogger(__name__)

# ======================================================
# LEXICON-BASED (LOCAL) ANALYZER
# (Энэ хэсэг шинээр нэмэгдсэн)
# ======================================================

# --- Лексикон үгийн жагсаалт ---
NEG = [
    "муу","гац","удаан","асуудал","нэмэгд","болохгүй","унана","алдаа",
    "таалагдсангүй","таалагдахгүй","тасрах","эвдэр","баланс дуус","унаад","уналаа","буруу"
]
POS = [
    "сайн","болсон","таалагд","баярлалаа","гоё","OK","ок","тасархай",
    "найдвартай","хурдан","амар","сайжир","дэвшил","баяр","баяртай"
]

TEXT_FIELDS = [
    "text","comment","comments","message","body","caption",
    "review","feedback","сэтгэгдэл","тайлбар"
]

# --- Лексикон туслах функцууд ---
def _pick_text_field(rec: Dict) -> str | None:
    for k in rec.keys():
        if k.lower() in TEXT_FIELDS:
            v = rec.get(k)
            if isinstance(v, str) and v.strip():
                return v
    return None

def _label(text: str) -> str:
    t = text.lower()
    p = sum(1 for w in POS if w in t)
    n = sum(1 for w in NEG if w in t)
    if n > p and n > 0:
        return "negative"
    if p > n and p > 0:
        return "positive"
    return "neutral"

# --- Лексикон Гол функц (Нэр нь __init__.py-д нийцүүлэгдсэн) ---
def analyze_sentiment(records: List[Dict]) -> Dict:
    """
    (ШИНЭ) Лексиконд суурилсан хурдан тооцоолуур.
    Олон тооны мөр (records) авч, дэлгэрэнгүй Dict (тоо, харьцаа) буцаана.
    """
    labels = []
    pos_ex, neg_ex, neu_ex = [], [], []
    n_text = 0

    for rec in records:
        txt = _pick_text_field(rec)
        if not txt:
            continue
        n_text += 1
        lab = _label(txt)
        labels.append(lab)
        if lab == "positive" and len(pos_ex) < 3:
            pos_ex.append(txt[:140])
        elif lab == "negative" and len(neg_ex) < 3:
            neg_ex.append(txt[:140])
        elif lab == "neutral" and len(neu_ex) < 3:
            neu_ex.append(txt[:140])

    c_pos = labels.count("positive")
    c_neg = labels.count("negative")
    c_neu = labels.count("neutral")
    tot = max(n_text, 1)

    return {
        "counts": {
            "positive": c_pos,
            "neutral": c_neu,
            "negative": c_neg,
            "total_texts": n_text,
        },
        "ratios": {
            "positive": round(100 * c_pos / tot, 1),
            "neutral": round(100 * c_neu / tot, 1),
            "negative": round(100 * c_neg / tot, 1),
        },
        "examples": {
            "positive": pos_ex,
            "negative": neg_ex,
            "neutral": neu_ex,
        },
        "has_text": n_text > 0,
    }


# ======================================================
# LLM-BASED (ADVANCED) ANALYZER
# ======================================================

# --- LLM туслах функцууд ---
def _extract_numbers(text: str) -> Dict[str, float]:
    """
    LLM markdown гаралтаас хувь хэмжээг автоматаар сугална.
    """
    if not text:
        return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}

    text = text.replace("％", "%")
    patt = re.compile(
        r"(эерэг|positive)\D*(\d{1,3}(?:\.\d+)?)%|"
        r"(саармаг|төв|neutral)\D*(\d{1,3}(?:\.\d+)?)%|"
        r"(сөрөг|negative)\D*(\d{1,3}(?:\.\d+)?)%",
        re.I,
    )
    out = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    for m in patt.finditer(text):
        g = m.groups()
        pairs: List[Tuple[str, str]] = [(g[i], g[i + 1]) for i in (0, 2, 4)]
        for label, val in pairs:
            if not label or not val:
                continue
            key = label.lower()
            num = float(val)
            if "эерэг" in key or "positive" in key:
                out["positive"] = num
            elif "саармаг" in key or "neutral" in key or "төв" in key:
                out["neutral"] = num
            elif "сөрөг" in key or "negative" in key:
                out["negative"] = num

    s = sum(out.values())
    if s > 0:
        k = 100.0 / s
        for t in out:
            out[t] = round(out[t] * k, 1)
    return out


def _mean_sentiments(items: List[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    keys = ("positive", "neutral", "negative")
    return {k: round(statistics.mean([d.get(k, 0.0) for d in items]), 1) for k in keys}


def _flatten_data(data: Union[str, dict, list]) -> str:
    """
    JSON эсвэл хүснэгтэн өгөгдлийг текст болгон хувиргана.
    """
    if isinstance(data, str):
        return data
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)

# ======================================================
# AI-BASED SENTIMENT ANALYZER (NEW)
# (ТАНЫ ХАМГИЙН СҮҮЛД НЭМСЭН ХЭСЭГ)
# ======================================================
def analyze_sentiment_ai(records: list[dict], meta: str = "") -> str:
    """
    Бүх сэтгэгдлийг нэгтгээд LLM-ээр (AI) дүгнэлт гаргана.
    """
    texts = []
    # (Шинэчилсэн: зөвхөн танигдсан текст талбарыг ашиглах)
    for r in records:
        txt = _pick_text_field(r)
        if txt and len(txt.strip()) > 3:
            texts.append(txt.strip())

    if not texts:
        return "⚠️ Файлд текстэн сэтгэгдэл олдсонгүй."

    # Хэт том файл бол эхний 800 мөрийг л ашиглана
    sample = "\n".join(texts[:800])
    prompt = (
        f"Доорх {len(texts)} хэрэглэгчийн сэтгэгдлийн ерөнхий хандлагыг дүгнэ. "
        f"Positive / Neutral / Negative хувийг гаргаж, 3–5 гол сэдэв, төлөөлөх ишлэл, дүгнэлт, зөвлөмжийг оруул. "
        f"Монгол хэлээр Markdown форматтайгаар бич.\n\n{sample}"
    )
    try:
        return ask_llm(prompt)
    except Exception as e:
        return f"⚠️ AI анализ амжилтгүй боллоо: {e}"


# --- LLM Гол Класс ---
class SentimentAnalyzer:
    """
    - CSV/Excel/JSON/текст/feedback файлууд уншина.
    - LLM ашиглан markdown тайлан гаргана.
    - Эерэг/саармаг/сөрөг оноог автоматаар тооцно.
    """
    def __init__(self) -> None:
        self.raw_texts: List[str] = []
        self.summary_text: Optional[str] = None
        self.scores: Dict[str, float] = {}

    # --------------------------- Load ---------------------------
    def load_from_file(self, path: str, text_cols: Optional[List[str]] = None, limit: int = 1000) -> List[str]:
        if not pd:
            raise ImportError("pandas шаардлагатай (`pip install pandas openpyxl`).")

        ext = path.lower().rsplit(".", 1)[-1]
        try:
            if ext in ("csv", "tsv"):
                df = pd.read_csv(path, sep="\t" if ext == "tsv" else ",", encoding="utf-8", on_bad_lines="skip")
            elif ext in ("xlsx", "xls"):
                df = pd.read_excel(path, dtype=str)
            elif ext == "json":
                df = pd.read_json(path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        except Exception as e:
            log.error("File read error: %s", e)
            raise

        # Автомат текст багана илрүүлэх
        if text_cols is None:
            candidates = ["comment", "comments", "text", "message", "content", "body", "review"]
            text_cols = [c for c in df.columns if c.lower() in candidates]
            if not text_cols:
                text_cols = [c for c in df.columns if str(df[c].dtype) == "object"]

        texts: List[str] = []
        for _, row in df[text_cols].fillna("").iterrows():
            joined = " ".join(str(row[c]).strip() for c in text_cols if str(row[c]).strip())
            if joined:
                texts.append(joined)
            if len(texts) >= limit:
                break

        self.raw_texts = texts
        return texts

    # --------------------------- Analyze ---------------------------
    def analyze(self, user_msg: str = "", meta: str = "") -> Dict[str, float]:
        if not self.raw_texts:
            raise ValueError("⚠️ Анализ хийх текст алга. Эхлээд load_from_file() эсвэл raw_texts-г тохируул.")

        # Том dataset-ийн хувьд зөвхөн эхний 500-г ашиглана.
        sample = "\n".join(self.raw_texts[:500])
        prompt = build_sentiment_prompt(user_msg or "Сэтгэгдлийн ерөнхий хандлагыг дүгнэ.", [(sample, meta)])
        try:
            self.summary_text = ask_llm(prompt)
            self.scores = _extract_numbers(self.summary_text)
        except Exception as e:
            log.error("Sentiment analysis failed: %s", e)
            self.summary_text = "⚠️ LLM анализ амжилтгүй боллоо."
            self.scores = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        return self.scores

    def summarize(self) -> str:
        return self.summary_text or "⚠️ Сэтгэгдлийн дүн хараахан үүсээгүй байна."

    def get_score(self) -> Dict[str, float]:
        return self.scores or {"positive": 0.0, "neutral": 0.0, "negative": 0.0}

# ======================================================
# QUICK ANALYSIS API (FOR SINGLE STRING)
# ======================================================
def analyze_sentiment_string(data: Union[str, dict, list]) -> str:
    """
    (ШИНЭ) LLM-д суурилсан хурдан тооцоолуур.
    Нэг (string) авч, "Positive" | "Neutral" | "Negative" гэсэн str буцаана.
    """
    try:
        analyzer = SentimentAnalyzer()

        if isinstance(data, (dict, list)):
            flat = _flatten_data(data)
            analyzer.raw_texts = [flat]
        else:
            analyzer.raw_texts = [str(data)]

        analyzer.analyze("Гол хандлагыг дүгнэж, эерэг/сөрөг байдлыг тооцоол.")
        scores = analyzer.get_score()
        pos, neu, neg = scores.get("positive", 0.0), scores.get("neutral", 0.0), scores.get("negative", 0.0)

        if pos >= max(neu, neg):
            return "Positive"
        if neg >= max(neu, pos):
            return "Negative"
        return "Neutral"
    except Exception as e:
        log.error("Quick sentiment analysis failed: %s", e)
        return "Neutral"

# ======================================================
# SELF TEST
# ======================================================
if __name__ == "__main__":
    # --- Test 1: Lexicon (Records) ---
    print("--- Testing Lexicon (Records) ---")
    recs = [
        {"comment": "Unitel-ийн үйлчилгээ маш сайн байна!"},
        {"comment": "Сүүлийн үед дата хурдан дуусдаг боллоо. муу."},
        {"comment": "Үнэ боломжийн, сүлжээ тасархай."},
    ]
    local_stats = analyze_sentiment(recs)
    print(json.dumps(local_stats, indent=2, ensure_ascii=False))

    # --- Test 2: LLM (AI Function) ---
    print("\n--- Testing LLM (AI Function) ---")
    ai_report = analyze_sentiment_ai(recs, meta="Test")
    print(ai_report)

    # --- Test 3: LLM (Class) ---
    print("\n--- Testing LLM (Class) ---")
    s = SentimentAnalyzer()
    s.raw_texts = [
        "Unitel-ийн үйлчилгээ маш сайн байна!",
        "Сүүлийн үед дата хурдан дуусдаг боллоо.",
        "Үнэ боломжийн, сүлжээ тасалдах нь бага.",
    ]
    print("Analyzing sentiment...")
    scores = s.analyze("Facebook comment сэтгэгдэлд анализ хий.")
    print("Scores:", scores)
    print(s.summarize())

    # --- Test 4: LLM (Quick String) ---
    print("\n--- Testing LLM (Quick String) ---")
    label = analyze_sentiment_string("маш муу гацаад байна")
    print(f"Label: {label}")