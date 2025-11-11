# ======================================================
# refresh_fb_token.py — Auto-refresh long-lived Facebook Page Token
# Author: Unitel AI Hub (2025)
# ======================================================

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

APP_ID = os.getenv("FB_APP_ID")
APP_SECRET = os.getenv("FB_APP_SECRET")
SHORT_TOKEN = os.getenv("FB_ACCESS_TOKEN")

GRAPH = "https://graph.facebook.com/v19.0"

def refresh_token():
    """
    Convert short-lived or expiring token to a long-lived token (60 days).
    Result: new token string printed and saved to .env (optional).
    """
    if not APP_ID or not APP_SECRET or not SHORT_TOKEN:
        print("⚠️ Missing FB_APP_ID, FB_APP_SECRET or FB_ACCESS_TOKEN in .env")
        return

    url = f"{GRAPH}/oauth/access_token"
    params = {
        "grant_type": "fb_exchange_token",
        "client_id": APP_ID,
        "client_secret": APP_SECRET,
        "fb_exchange_token": SHORT_TOKEN,
    }

    res = requests.get(url, params=params)
    if res.ok:
        data = res.json()
        new_token = data.get("access_token")
        print("✅ New long-lived token:")
        print(new_token)

        # Optional: automatically update your .env file
        update_env(".env", "FB_ACCESS_TOKEN", new_token)
        print("✅ .env file updated successfully.")
    else:
        print(f"❌ Failed to refresh token: {res.status_code}")
        print(res.text)

def update_env(filepath, key, new_value):
    """Update or add a key=value pair inside .env file."""
    lines = []
    found = False
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(f"{key}="):
                    lines.append(f"{key}={new_value}\n")
                    found = True
                else:
                    lines.append(line)
    if not found:
        lines.append(f"{key}={new_value}\n")

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)

if __name__ == "__main__":
    refresh_token()
