# ======================================================
# processor/file_handler.py
# Universal file reader & normalizer for AI processing
# Author: Unitel AI Hub (2025 edition)
# ======================================================

from __future__ import annotations
import os, json, logging
from typing import Any, Dict, List, Union, Tuple
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document

log = logging.getLogger(__name__)

# Optional Mongo logging (safe import)
try:
    from database.mongo import save_raw_data
except Exception:
    def save_raw_data(*args, **kwargs):  # safe fallback
        return None

__all__ = [
    "detect_type", "read_file", "summarize_dataframe",
    "extract_text_from_file",
    "extract_records_from_file"  # <-- ШИНЭЭР НЭМЭГДСЭН
]
# ======================================================
# SMART FILE PURPOSE DETECTION
# ======================================================

def detect_file_purpose(df: pd.DataFrame) -> str:
    """Return file purpose such as sentiment, influencer, or general."""
    cols = [c.lower() for c in df.columns]
    if any(k in cols for k in ["comment", "feedback", "review", "message", "text"]):
        return "sentiment"
    elif any(k in cols for k in ["views", "video views", "reach", "impressions", "likes", "comments", "followers", "shares"]):
        return "influencer"
    elif any(k in cols for k in ["spend", "revenue", "sales", "cost"]):
        return "financial"
    else:
        return "general"


def summarize_influencer_metrics(df: pd.DataFrame) -> str:
    """Summarize influencer or campaign performance data."""
    out = []
    total_videos = len(df)
    out.append(f"Нийт бичлэгийн тоо: {total_videos}")

    for col in df.columns:
        if "view" in col.lower():
            out.append(f"Нийт үзэлт: {df[col].astype(float).sum():,.0f}")
        if "comment" in col.lower():
            out.append(f"Нийт коммент: {df[col].astype(float).sum():,.0f}")
        if "like" in col.lower():
            out.append(f"Нийт лайк: {df[col].astype(float).sum():,.0f}")
        if "share" in col.lower():
            out.append(f"Нийт хуваалцалт: {df[col].astype(float).sum():,.0f}")

    # Top performer
    metric_cols = [c for c in df.columns if any(k in c.lower() for k in ["view", "impression", "reach"])]
    if metric_cols:
        top_idx = df[metric_cols[0]].astype(float).idxmax()
        top_row = df.loc[top_idx]
        out.append(f"Хамгийн өндөр үзэлттэй бичлэг: “{top_row.get('Title', top_row.get('Caption', '—'))}”")

    return "\n".join(out)


def summarize_general_dataframe(df: pd.DataFrame) -> str:
    """Fallback summary for general structured data."""
    text = f"Мөрийн тоо: {len(df)} | Багана: {len(df.columns)}\n"
    text += f"Баганууд: {', '.join(df.columns[:10])}"
    return text

# -------------------- CONFIG --------------------
MAX_ROWS = 5000          # dataframe sampling limit
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# FILE DETECTION
# ======================================================
def detect_type(filepath: str) -> str:
    """Return simplified filetype string."""
    ext = Path(filepath).suffix.lower()
    if ext in [".csv", ".tsv"]: return "csv"
    if ext in [".xls", ".xlsx"]: return "excel"
    if ext == ".json": return "json"
    if ext == ".pdf": return "pdf"
    if ext == ".docx": return "docx"
    if ext in [".html", ".htm"]: return "html"
    return "text"

# ======================================================
# FILE READERS
# ======================================================
def _sample_df(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) > MAX_ROWS:
        return df.sample(MAX_ROWS, random_state=42)
    return df

def read_csv(filepath: str) -> List[Dict[str, Any]]:
    try:
        sep = "," if filepath.endswith(".csv") else "\t"
        df = pd.read_csv(filepath, sep=sep, encoding="utf-8", on_bad_lines="skip")
        df = _sample_df(df)
        return df.to_dict(orient="records")
    except Exception as e:
        log.error("CSV read error: %s", e)
        return []

def read_excel(filepath: str) -> List[Dict[str, Any]]:
    try:
        df = pd.read_excel(filepath, dtype=str)
        df = _sample_df(df)
        return df.to_dict(orient="records")
    except Exception as e:
        log.error("Excel read error: %s", e)
        return []

def read_json(filepath: str) -> Any:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [data] if isinstance(data, dict) else data
    except Exception as e:
        log.error("JSON read error: %s", e)
        return []

def read_pdf(filepath: str) -> str:
    texts = []
    try:
        reader = PdfReader(filepath)
        for p in reader.pages:
            txt = p.extract_text()
            if txt:
                texts.append(txt.strip())
    except Exception as e:
        log.error("PDF read error: %s", e)
    return "\n".join(texts)

def read_docx(filepath: str) -> str:
    try:
        doc = Document(filepath)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        log.error("DOCX read error: %s", e)
        return ""

def read_html(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(" ", strip=True)
    except Exception as e:
        log.error("HTML read error: %s", e)
        return ""

def read_text(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        log.error("TEXT read error: %s", e)
        return ""

# ======================================================
# MAIN LOADER
# ======================================================
def read_file(filepath: str) -> Union[str, List[Dict[str, Any]]]:
    """
    Read any file and return structured data.
    If structured (csv/json/xlsx) -> List[dict]
    Else -> raw text
    """
    ftype = detect_type(filepath)
    log.info(f"📂 Reading file: {filepath} ({ftype})")

    if ftype == "csv":
        data = read_csv(filepath)
    elif ftype == "excel":
        data = read_excel(filepath)
    elif ftype == "json":
        data = read_json(filepath)
    elif ftype == "pdf":
        data = read_pdf(filepath)
    elif ftype == "docx":
        data = read_docx(filepath)
    elif ftype == "html":
        data = read_html(filepath)
    else:
        data = read_text(filepath)

    # Log to MongoDB (optional)
    try:
        save_raw_data(source=f"upload_{ftype}", data=data)
    except Exception:
        pass

    return data

# ======================================================
# STRUCTURED DATA EXTRACTOR (Таны нэмэхийг хүссэн хэсэг)
# ======================================================
TABULAR_EXT = (".csv", ".tsv", ".xlsx", ".xls")

def extract_records_from_file(path: str) -> tuple[list[dict], str]:
    """
    Excel/CSV файлыг list[dict] болгон уншина. (эхний sheet / RAW sheet)
    """
    p = Path(path)
    meta = f"{p.name}"
    ext = p.suffix.lower()

    try:
        if ext in (".xlsx", ".xls"):
            xls = pd.ExcelFile(p)
            sheet = "RAW" if "RAW" in xls.sheet_names else xls.sheet_names[0]
            df = pd.read_excel(xls, sheet_name=sheet)
            return (df.fillna("").to_dict(orient="records"), f"{p.name} [{sheet}]")

        if ext in (".csv", ".tsv"):
            sep = "," if ext == ".csv" else "\t"
            df = pd.read_csv(p, sep=sep, low_memory=False)
            return (df.fillna("").to_dict(orient="records"), meta)
            
    except Exception as e:
        return ([], f"{meta} [read_error: {e}]")

    # Таблич биш бол хоосон буцаана
    return ([], meta)

# ======================================================
# LLM-FRIENDLY EXTRACTOR (Үндсэн кодонд байсан хэсэг)
# ======================================================
def extract_text_from_file(path: str) -> Tuple[str, str]:
    """
    Return (content_text, meta_string) for LLM prompts.
    If structured (list of dicts), flatten to a compact text format.
    """
    p = Path(path)
    ftype = detect_type(path)
    data = read_file(path)

    if isinstance(data, list):  # structured (table-like)
        if not data:
            text = "(хоосон хүснэгт)"
            rows = 0
            cols = 0
            cols_used: List[str] = []
        else:
            rows = len(data)
            all_cols = list({k for row in data for k in row.keys()})
            cols = len(all_cols)
            cols_used = all_cols[:10]
            lines = []
            for r in data[:1000]:
                parts = []
                for c in cols_used:
                    v = str(r.get(c, "")).replace("\n", " ").strip()
                    if len(v) > 200:
                        v = v[:200] + "…"
                    parts.append(f"{c}={v}")
                lines.append(" | ".join(parts))
            text = "\n".join(lines)
        meta = f"[TABLE:{p.suffix}] {p.name} rows={rows} cols={cols} preview_cols={','.join(cols_used)}"

    else:  # plain text
        raw = str(data or "")
        if len(raw) > 120_000:
            raw = raw[:120_000] + "\n…(truncated)"
        text = raw
        meta = f"[{ftype.upper()}:{p.suffix}] {p.name}"

    return text, meta

# ======================================================
# DATA NORMALIZATION
# ======================================================
def summarize_dataframe(records: List[Dict[str, Any]]) -> str:
    """Generate quick textual summary from a list of dict rows."""
    if not records:
        return "Хоосон өгөгдөл байна."
    df = pd.DataFrame(records)
    summary = {
        "мөрийн тоо": len(df),
        "баганын тоо": len(df.columns),
        "баганууд": list(df.columns),
    }
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text = f"Мөрийн тоо: {summary['мөрийн тоо']}\n"
    text += f"Баганууд: {', '.join(summary['баганууд'])}\n"
    if numeric_cols:
        text += f"Тоон баганууд: {', '.join(numeric_cols)}\n"
    return text

# ======================================================
# QUICK TEST
# ======================================================
if __name__ == "__main__":
    test = "sample.xlsx"
    if os.path.exists(test):
        content, meta = extract_text_from_file(test)
        print(meta)
        print(content[:800])
    else:
        print("⚠️ Test file not found.")