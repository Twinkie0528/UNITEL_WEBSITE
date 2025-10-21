# processor.py — File-aware LLM assistant (clean markdown outputs)
# Author: Unitel AI Hub

from __future__ import annotations
import os, re, json, math, pickle, random, csv
from typing import List, Optional, Tuple, Dict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---------------- .env ----------------
from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

# ---------------- Optional local intent model (kept but optional) ----------------
DISABLE_LOCAL = os.getenv("UNITEL_DISABLE_LOCAL", "1") == "1"  # default: OFF local model

lemmatizer = None
try:
    from keras.models import load_model
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import wordpunct_tokenize
    lemmatizer = WordNetLemmatizer()
except Exception:
    pass

def _p(name: str) -> Path: return BASE_DIR / name

# Try to load local assets only if available
intents = None; words = None; classes = None; model = None
try:
    if not DISABLE_LOCAL:
        intents = json.loads((_p("job_intents.json")).read_text(encoding="utf-8"))
        with open(_p("words.pkl"), "rb") as fh: words = pickle.load(fh)
        with open(_p("classes.pkl"), "rb") as fh: classes = pickle.load(fh)
        model = load_model(_p("chatbot_model.h5"))
except Exception:
    model = None

def clean_up_sentence(sentence: str):
    toks = wordpunct_tokenize(sentence) if lemmatizer else sentence.split()
    return [lemmatizer.lemmatize(w.lower()) if lemmatizer else w.lower() for w in toks]

def bow(sentence, vocab):
    sw = clean_up_sentence(sentence)
    return [1 if w in sw else 0 for w in vocab] if vocab else []

def predict_class(sentence, threshold=0.4):
    if model is None or words is None or classes is None:
        return []
    import numpy as np
    res = model.predict(np.array([bow(sentence, words)]), verbose=0)[0]
    ranked = [[i, r] for i, r in enumerate(res) if r > threshold]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[i], "p": float(p)} for i, p in ranked]

# ---------------- OpenAI client ----------------
from openai import OpenAI, OpenAIError
_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
_client = OpenAI(api_key=_OPENAI_KEY) if _OPENAI_KEY else None

MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
FALLBACK = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini")
DISABLE_FALLBACK = os.getenv("OPENAI_DISABLE_FALLBACK", "0") == "1"
MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "4000"))  # generous but safe

SYSTEM_PROMPT = os.getenv(
    "UNITEL_SYSTEM_PROMPT",
    "Та Unitel Assistant. Бүх хариултаа **markdown** форматтай, цэгцтэй өг. "
    "Эхэнд товч гарчиг, дараа нь bullet/тоо бүхий жагсаалт, эцэст нь богино дүгнэлт гарга. "
    "Илүүдэл чимэглэл, эмодзи, урт нуршил бүү ашигла."
)

def _call_responses(system_text: str, user_text: str) -> str:
    r = _client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": [{"type":"input_text","text": system_text}]},
            {"role": "user",   "content": [{"type":"input_text","text": user_text}]},
        ],
        max_output_tokens=MAX_TOKENS,
    )
    return (r.output_text or "").strip()

def _call_chat(system_text: str, user_text: str) -> str:
    r = _client.chat.completions.create(
        model=FALLBACK,
        messages=[{"role":"system","content":system_text},
                  {"role":"user","content":user_text}],
        temperature=0.2,
        max_tokens=MAX_TOKENS,
    )
    return (r.choices[0].message.content or "").strip()

def ask_openai(user_msg: str) -> str:
    if not _client: return "⚠️ OpenAI түлхүүр тохируулагдаагүй байна."
    try:
        return _call_responses(SYSTEM_PROMPT, user_msg)
    except OpenAIError as e:
        msg = str(e)
        bad_request = ("400" in msg) or ("invalid_request_error" in msg) or ("Unsupported parameter" in msg)
        if bad_request and not DISABLE_FALLBACK:
            try: return _call_chat(SYSTEM_PROMPT, user_msg)
            except OpenAIError as e2:
                em = str(e2)
                if "insufficient_quota" in em: return "⚠️ Квот хүрсэн байна."
                if "rate_limit" in em: return "⚠️ Одоогоор хүсэлт их байна. Дараа дахин оролдоно уу."
                return f"⚠️ LLM fallback алдаа: {e2.__class__.__name__}"
        if "insufficient_quota" in msg: return "⚠️ Квот хүрсэн байна."
        if "rate_limit" in msg: return "⚠️ Одоогоор хүсэлт их байна. Дараа дахин оролдоно уу."
        return f"⚠️ LLM алдаа: {e.__class__.__name__}"

# ---------------- File extraction utilities ----------------
# Optional deps: pandas, openpyxl, PyPDF2, python-docx, beautifulsoup4
def _import_or_none(mod):
    try:
        return __import__(mod)
    except Exception:
        return None

pd = _import_or_none("pandas")
PyPDF2 = _import_or_none("PyPDF2")
docx = _import_or_none("docx")
bs4 = _import_or_none("bs4")

TEXT_EXT  = {".txt", ".log", ".md"}
TABULAR_EXT = {".csv", ".tsv", ".xlsx"}
JSON_EXT  = {".json"}
PDF_EXT   = {".pdf"}
DOCX_EXT  = {".docx"}
HTML_EXT  = {".html", ".htm"}

# Heuristics: which columns likely contain text
LIKELY_TEXT_COLS = ["comment", "comments", "text", "message", "content", "body", "review"]

def _read_small_text_file(p: Path, limit_chars=120_000) -> str:
    try:
        s = p.read_text(encoding="utf-8", errors="ignore")
        return s[:limit_chars]
    except Exception as e:
        return f"[read_text_error:{e}]"

def _read_json(p: Path, limit=500):
    try:
        data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        out = []
        def walk(x, path=""):
            if isinstance(x, dict):
                for k,v in x.items(): walk(v, f"{path}.{k}" if path else k)
            elif isinstance(x, list):
                for i,v in enumerate(x[:limit]): walk(v, f"{path}[{i}]")
            else:
                if isinstance(x, (str,int,float,bool)): out.append(f"{path}: {x}")
        walk(data)
        return "\n".join(out[:4000])
    except Exception as e:
        return f"[json_read_error:{e}]"

def _read_pdf(p: Path, limit_pages=20) -> str:
    if not PyPDF2: return "[pdf_reader_missing: pip install PyPDF2]"
    try:
        reader = PyPDF2.PdfReader(str(p))
        pages = min(len(reader.pages), limit_pages)
        text = []
        for i in range(pages):
            try:
                text.append(reader.pages[i].extract_text() or "")
            except Exception:
                continue
        return "\n".join(text)
    except Exception as e:
        return f"[pdf_read_error:{e}]"

def _read_docx(p: Path) -> str:
    if not docx: return "[docx_reader_missing: pip install python-docx]"
    try:
        d = docx.Document(str(p))
        return "\n".join([para.text for para in d.paragraphs if para.text])
    except Exception as e:
        return f"[docx_read_error:{e}]"

def _read_html(p: Path, limit_chars=120_000) -> str:
    if not bs4:
        return _read_small_text_file(p, limit_chars)
    try:
        from bs4 import BeautifulSoup
        s = p.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(s, "html.parser")
        return (soup.get_text(separator="\n") or "")[:limit_chars]
    except Exception as e:
        return f"[html_read_error:{e}]"

def _read_tabular(p: Path) -> Tuple[str, Optional[str]]:
    """
    Returns (joined_text, hint)
    - Heuristically select text-like columns; if not found, join all cols.
    - Sample up to N rows to keep LLM tokens in check.
    """
    if not pd:
        return ("[pandas_missing: pip install pandas openpyxl]", None)

    ext = p.suffix.lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(p, encoding="utf-8", on_bad_lines="skip")
        elif ext == ".tsv":
            df = pd.read_csv(p, sep="\t", encoding="utf-8", on_bad_lines="skip")
        elif ext == ".xlsx":
            df = pd.read_excel(p, dtype=str)  # openpyxl auto
        else:
            return (f"[unsupported_tabular:{ext}]", None)
    except Exception as e:
        return (f"[tabular_read_error:{e}]", None)

    # choose likely text columns
    cols = [c for c in df.columns]
    text_cols = [c for c in cols if c.lower() in LIKELY_TEXT_COLS]
    if not text_cols:
        obj_cols = [c for c in cols if str(df[c].dtype) == "object"]
        text_cols = obj_cols or cols

    # sample rows
    N = len(df)
    if N <= 400:
        sample = df
    else:
        head = df.head(200)
        tail = df.tail(100)
        mid  = df.sample(min(100, N-300), random_state=7) if N > 300 else df
        sample = pd.concat([head, mid, tail]).drop_duplicates().head(500)

    rows = []
    for _, r in sample[text_cols].fillna("").iterrows():
        line = " | ".join(str(r[c]).strip() for c in text_cols if str(r[c]).strip())
        if line: rows.append(line)

    hint = f"rows={N}, sampled={len(rows)}, columns_used={', '.join(map(str, text_cols[:6]))}"
    return ("\n".join(rows[:6000]), hint)

def extract_text_from_file(path: str) -> Tuple[str, str]:
    """
    Return (content, meta) where:
      - content: extracted text (limited)
      - meta: brief description for the prompt
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in TABULAR_EXT:
        content, hint = _read_tabular(p)
        meta = f"[TABULAR:{ext}] {p.name} {hint or ''}".strip()
        return content, meta
    if ext in TEXT_EXT:
        return _read_small_text_file(p), f"[TEXT:{ext}] {p.name}"
    if ext in JSON_EXT:
        return _read_json(p), f"[JSON] {p.name}"
    if ext in PDF_EXT:
        return _read_pdf(p), f"[PDF] {p.name}"
    if ext in DOCX_EXT:
        return _read_docx(p), f"[DOCX] {p.name}"
    if ext in HTML_EXT:
        return _read_html(p), f"[HTML] {p.name}"

    # Fallback: try read as text
    return _read_small_text_file(p), f"[UNKNOWN:{ext}] {p.name}"

# ---------------- High-level task builders ----------------
SENTIMENT_INSTRUCTIONS = """
Та өгөгдлөөс сэтгэл хөдлөлийн тойм гаргана уу.
**ГАРАЛТ ЗААВАЛ markdown** форматтай, доорх **яг энэ бүтэцтэй** байна:

## Нийт дүн
- Эерэг: X%
- Саармаг/Төвийн: Y%
- Сөрөг: Z%

## Гол сэдвүүд (5 хүртэл) — дугаарласан жагсаалт
1. <Сэдвийн нэр> — 1–2 өгүүлбэр тайлбар.
   - Хөндсөн эзлэх хувь: ~A%
   - Сэтгэл хөдлөл: эерэг ~B% • төв ~C% • сөрөг ~D%
2. <…>
3. <…>

## Төлөөлөх ишлэлүүд (3–6 богино)
- “…” 
- “…” 
- “…”

## Дүгнэлт / Зөвлөмж
- 2–4 bullet байдлаар actionable санал.

**Тайлбар**  
- “Төвийн” гэж мэдээллийн шинжтэй, эерэг/сөрөг үнэлгээ тодорхой бус, асуулт маягийн агуулгыг ойлгоно.
- Ойлгомжгүй урт догол мөр, эмодзи, илүү чимэглэл хэрэглэхгүй.
"""

def build_sentiment_prompt(user_msg: str, file_blobs: List[Tuple[str,str]]) -> str:
    parts = [SENTIMENT_INSTRUCTIONS.strip()]
    if user_msg:
        parts.append(f"### Хэрэглэгчийн шаардлага\n{user_msg.strip()}")
    for text, meta in file_blobs:
        parts.append(f"### Файл: {meta}\n{text[:120000]}")
    return "\n\n".join(parts)

GENERAL_INSTRUCTIONS = """
Та монгол хэлээр товч, **markdown** хэлбэрээр хариул. Бүтэц:
- `## Тойм` — 2–3 өгүүлбэр
- `## Гол санаа` — 3–7 bullet
- `## Дүгнэлт` — 1 догол мөр
(Хэрэв хүсвэл жижиг хүснэгт оруулж болно.)
"""

def build_general_prompt(user_msg: str, file_blobs: List[Tuple[str,str]]) -> str:
    parts = [GENERAL_INSTRUCTIONS.strip(), f"## Асуулт\n{user_msg.strip() or 'Доорх материалд үндэслэн хариул.'}"]
    for text, meta in file_blobs:
        parts.append(f"## Ашиглах материал — {meta}\n{text[:100000]}")
    return "\n\n".join(parts)

# ---------------- Public entrypoint ----------------
def chatbot_response(msg: str, files: Optional[List[str]] = None) -> str:
    """
    Unified chat entry:
    - If files attached: extract text; if the intent seems 'sentiment' or data looks like comments table,
      build a sentiment prompt; else general prompt with materials.
    - If no file: use LLM general Q&A (and keep markdown).
    """
    msg = (msg or "").strip()
    files = files or []

    if not _client:
        return "⚠️ OpenAI түлхүүр тохируулагдаагүй байна. .env дахь OPENAI_API_KEY-г тохируулна уу."

    # Extract materials
    blobs: List[Tuple[str,str]] = []
    for f in files:
        try:
            content, meta = extract_text_from_file(f)
            if content and not str(content).startswith("[*_error"):
                blobs.append((content, meta))
        except Exception as e:
            blobs.append((f"[extract_error:{e}]", f"[{Path(f).name}]"))

    # Heuristic: sentiment?
    lower = msg.lower()
    wants_sentiment = any(k in lower for k in ["sentiment", "сентимент", "сэтгэгдэл", "эерэг", "сөрөг", "саармаг"])
    looks_like_comments = any(
        ("tabular" in b[1].lower() or any(ext in b[1].lower() for ext in [".xlsx", ".csv", ".tsv"]))
        for b in blobs
    )

    if blobs and (wants_sentiment or looks_like_comments):
        prompt = build_sentiment_prompt(msg or "Facebook/Univision сэтгэгдлүүдийн дүн шинжилгээ хий.", blobs)
        return ask_openai(prompt)

    if blobs:
        prompt = build_general_prompt(msg or "Доорх материалд үндэслэн дүгнэлт гарга.", blobs)
        return ask_openai(prompt)

    # No files → plain chat
    prompt = build_general_prompt(msg or "Сайн уу", [])
    return ask_openai(prompt)
