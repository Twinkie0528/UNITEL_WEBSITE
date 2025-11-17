# ============================================================
# processor/intent_classifier.py
# GPT-based Intent Classifier (Replaces TensorFlow model)
# (Live-Reload хувилбар)
# ============================================================

import json
from pathlib import Path
from typing import Optional, Dict, List
from .llm_client import ask_llm  # GPT client

BASE_DIR = Path(__file__).resolve().parent.parent

# ❌ УСТГАСАН БЛОК: Энд байсан статик intent load-ийг устгасан.

# ------------------------------------------------------------
# DYNAMIC INTENT LOADER (Live reload every call)
# (✅ Шинээр нэмэгдсэн)
# ------------------------------------------------------------
def load_intents() -> dict:
    """Always reload job_intents.json dynamically."""
    try:
        return json.loads(
            (BASE_DIR / "job_intents.json").read_text(encoding="utf-8")
        )
    except Exception:
        return {"intents": []}


# ------------------------------------------------------------
# Build prompt for GPT classifier
# ------------------------------------------------------------
def _build_classifier_prompt(message: str, JOB_INTENTS: dict) -> str:
    """
    ✅ Өөрчлөлт:
    Энэ функц рүү 'JOB_INTENTS'-г гаднаас (classify_intent)-ээс дамжуулдаг болсон.
    """
    intents_list = [
        {
            "tag": it.get("tag"),
            "patterns": it.get("patterns", []),
            "contextual_keywords": it.get("contextual_keywords", [])
        }
        for it in JOB_INTENTS.get("intents", [])
    ]

    return f"""
You are an INTENT CLASSIFIER.

Your task:
- Read the user message.
- Compare it with defined intents.
- Choose EXACTLY ONE intent tag.
- If no match is strong → return "none".

Respond ONLY with JSON in this format:

{{
  "tag": "<best_intent_tag_or_none>"
}}

------------------------
INTENT DEFINITIONS:
{json.dumps(intents_list, ensure_ascii=False, indent=2)}
------------------------

USER MESSAGE:
"{message}"
"""


# ------------------------------------------------------------
# GPT Classifier (Main function)
# ------------------------------------------------------------
def classify_intent(message: str) -> str:
    """
    Returns:
        tag (str): intent tag from job_intents.json or "none"
    """
    if not message or not message.strip():
        return "none"

    # ✅ Өөрчлөлт: JSON-г энд динамикаар ачаална
    JOB_INTENTS = load_intents()   # <--- LIVE RELOAD
    
    # ✅ Өөрчлөлт: Ачаалсан intent-ээ prompt builder рүү дамжуулна
    prompt = _build_classifier_prompt(message, JOB_INTENTS)

    try:
        raw = ask_llm(prompt)
        data = json.loads(raw)
        tag = data.get("tag", "none")
        return tag
    except Exception:
        return "none"


# ------------------------------------------------------------
# (Optional) Return a response from job_intents.json
# ------------------------------------------------------------
def get_intent_response(tag: str) -> Optional[str]:
    # ✅ Өөрчлөлт: JSON-г энд мөн динамикаар ачаална
    JOB_INTENTS = load_intents() # <--- LIVE RELOAD

    for it in JOB_INTENTS.get("intents", []):
        if it.get("tag") == tag:
            responses = it.get("responses") or []
            if responses:
                import random
                return random.choice(responses)
    return None