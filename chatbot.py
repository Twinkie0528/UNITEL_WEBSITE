import json
import os
from openai import OpenAI

# Load environment key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Load job intents
with open("job_intents.json", "r", encoding="utf-8") as f:
    INTENTS = json.load(f)["intents"]

def classify_intent(user_message: str) -> dict:
    """
    Classifies user message into one of the intents using GPT classifier
    """
    tags = [i["tag"] for i in INTENTS]

    prompt = f"""
You are an intent classification model.

User message: "{user_message}"

Possible intents:
{tags}

Return only JSON:
{{
  "tag": "...",
  "confidence": 0-1
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        result = response.choices[0].message.content
        return json.loads(result)
    except:
        return {"tag": "unknown", "confidence": 0.0}
