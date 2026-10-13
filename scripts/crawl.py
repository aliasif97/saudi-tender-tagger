#!/usr/bin/env python3
"""Polite unfiltered crawl of Etimad's public visitor listing, newest first.
Same endpoint and method as etimad-watch/family-scan.py (curl_cffi, no keyword).
Resume-safe: each page's raw JSON is cached in data/raw/page_NNNN.json.
Rate: one request per MIN_GAP seconds (default 8.6, i.e. ~7/min, which is what
Etimad tolerates; stricter than the 1 per 2 s ceiling). Stops on repeated
non-JSON responses (bot wall) and does not try to get around it."""
import json, os, sys, time
from curl_cffi import requests
BASE = "https://tenders.etimad.sa/Tender/AllSupplierTendersForVisitorAsync"
H = {"X-Requested-With": "XMLHttpRequest", "Referer": "https://tenders.etimad.sa/Tender/AllTendersForVisitor"}
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
MIN_GAP = float(os.environ.get("MIN_GAP", "8.6"))
MAX_PAGES = int(os.environ.get("MAX_PAGES", "260"))
MAX_SECONDS = int(os.environ.get("MAX_SECONDS", "2700"))
os.makedirs(RAW, exist_ok=True)
t0 = time.time(); last = 0; fails = 0
for pg in range(1, MAX_PAGES + 1):
    fn = os.path.join(RAW, f"page_{pg:04d}.json")
    if os.path.exists(fn): continue
    if time.time() - t0 > MAX_SECONDS: print("time cap reached"); break
    qs = ("MultipleSearch=&TenderCategory=&ReferenceNumber=&TenderNumber=&AgencyCode=&PublishDateId=&TenderTypeId="
          "&TenderActivityId=&SortDirection=DESC&Sort=SubmitionDate&PageSize=24&PageNumber=%d&IsSearch=false" % pg)
    ok = False
    for attempt in range(3):
        wait = MIN_GAP - (time.time() - last)
        if wait > 0: time.sleep(wait)
        last = time.time()
        try:
            r = requests.get(BASE + "?" + qs, impersonate="safari", timeout=45, headers=H)
            j = r.json()
            if not j.get("data"): print(f"page {pg}: empty, end of listing"); sys.exit(0)
            json.dump(j, open(fn, "w"), ensure_ascii=False); ok = True; fails = 0
            print(f"page {pg} ok ({len(j['data'])}) total={j.get('totalCount')}", flush=True); break
        except Exception as e:
            fails += 1; print(f"page {pg} attempt {attempt+1} failed: {str(e)[:80]}", flush=True)
            time.sleep(15 * (attempt + 1))
    if not ok and fails >= 6:
        print("BLOCKED: repeated failures, stopping"); sys.exit(2)
    if not ok: continue
