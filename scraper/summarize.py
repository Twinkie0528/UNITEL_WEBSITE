"""Legacy summarizer.

Exports are generated on-demand by the Flask app via /report/ads.(tsv|xlsx).
This script is kept for backwards compatibility only.
"""

if __name__ == "__main__":
    print("Use /report/ads.tsv or /report/ads.xlsx from the Flask app for fresh exports.")
