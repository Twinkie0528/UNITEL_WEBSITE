# fb_scraper.py
import os, csv, datetime as dt, requests

GRAPH = "https://graph.facebook.com/v19.0"
PAGE_ID = os.getenv("FB_PAGE_ID")                 # таны Page ID
ACCESS_TOKEN = os.getenv("FB_PAGE_TOKEN")         # Page access token (long-lived)

def gget(path, **params):
    params["access_token"] = ACCESS_TOKEN
    r = requests.get(f"{GRAPH}/{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_posts(since: str, until: str, limit=100):
    """
    since/until: 'YYYY-MM-DD' (UTC)
    """
    fields = "id,message,created_time,permalink_url"
    params = dict(fields=fields, limit=limit, since=since, until=until)
    url = f"{PAGE_ID}/posts"
    out = []
    while url:
        r = requests.get(f"{GRAPH}/{url.split('graph.facebook.com/')[1]}", params={**params, "access_token": ACCESS_TOKEN}) \
            if "graph.facebook.com" in url else requests.get(f"{GRAPH}/{url}", params={**params, "access_token": ACCESS_TOKEN})
        r.raise_for_status()
        data = r.json()
        out += data.get("data", [])
        url = data.get("paging", {}).get("next")
        params = {}  # next URL дотор бүх параметр орчихсон байдаг
    return out

def fetch_all_comments(post_id, limit=100):
    fields = "id,message,from,created_time,like_count,comment_count,permalink_url,parent"
    url = f"{post_id}/comments"
    params = dict(fields=fields, limit=limit, summary="true", access_token=ACCESS_TOKEN)
    out = []
    while url:
        r = requests.get(f"{GRAPH}/{url}", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        out += data.get("data", [])
        url = data.get("paging", {}).get("next")
        params = {}
    return out

def export_posts_and_comments(since="2025-01-01", until="2025-12-31", outdir="static/fb"):
    os.makedirs(outdir, exist_ok=True)
    posts = fetch_posts(since, until)
    # Posts CSV
    p_csv = os.path.join(outdir, "posts.csv")
    with open(p_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["post_id","message","created_time","permalink_url"])
        for p in posts:
            w.writerow([p.get("id"), (p.get("message") or "").replace("\n"," "), p.get("created_time"), p.get("permalink_url")])
    # Comments CSV (all posts)
    c_csv = os.path.join(outdir, "comments.csv")
    with open(c_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["post_id","comment_id","author_id","author_name","message","created_time","like_count","comment_count","permalink_url","parent_id"])
        for p in posts:
            pid = p["id"]
            for c in fetch_all_comments(pid):
                frm = c.get("from") or {}
                w.writerow([
                    pid, c.get("id"), frm.get("id"), frm.get("name"),
                    (c.get("message") or "").replace("\n"," "),
                    c.get("created_time"), c.get("like_count"), c.get("comment_count"),
                    c.get("permalink_url"), (c.get("parent") or {}).get("id")
                ])
    return p_csv, c_csv

if __name__ == "__main__":
    # Жишээ: сүүлийн 30 хоног
    today = dt.date.today()
    since = (today - dt.timedelta(days=30)).isoformat()
    until = today.isoformat()
    print(export_posts_and_comments(since, until))
