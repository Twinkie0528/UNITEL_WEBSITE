import os, time, requests
from apscheduler.schedulers.background import BackgroundScheduler

sched = BackgroundScheduler()

# Жишээ: 30 минут тутам dummy ingest хийнэ (туршилтад)
def heartbeat():
    try:
        requests.post("http://localhost:8888/ads/api/ingest", json={
            "ad_id": "demo_banner",
            "site": "news.mn" "gogo.mn" "ikon.mn",
            "status": "active"
        }, timeout=5)
    except Exception as e:
        print("heartbeat failed:", e)

if __name__ == "__main__":
    sched.add_job(heartbeat, "interval", minutes=30, id="hb")
    sched.start()
    print("Scheduler started. Ctrl+C to exit.")
    try:
        while True: time.sleep(60)
    except KeyboardInterrupt:
        sched.shutdown()
