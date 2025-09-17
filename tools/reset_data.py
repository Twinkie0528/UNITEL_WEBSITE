from __future__ import annotations

import os
import sys
from urllib.parse import urlencode

import requests

BASE = os.getenv("RESET_BASE", "http://127.0.0.1:8888").rstrip("/")
TOKEN = os.getenv("RESET_TOKEN")

if not TOKEN:
    print("RESET_TOKEN is required")
    sys.exit(1)

url = f"{BASE}/tools/reset?{urlencode({'token': TOKEN})}"
print(f"Calling {url} ...")
resp = requests.get(url, timeout=30)
print(resp.status_code, resp.text)
resp.raise_for_status()
print("Reset completed.")
