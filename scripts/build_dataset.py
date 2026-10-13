#!/usr/bin/env python3
"""Merge local sources + crawl cache into data/tenders_all.csv (tender fields only).
Usage: build_dataset.py [--base-only]
Priority when a ref repeats: crawl > last-scan > awards.db > xlsx (first non-empty field wins)."""
import csv, glob, json, os, re, sqlite3, sys, unicodedata
from collections import Counter
import openpyxl
W = "/Users/aliasif/.claude/etimad-watch/"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
base_only = "--base-only" in sys.argv
FIELDS = ["ref", "title", "agency", "branch", "activity", "ttype", "publish_date", "source"]
def clean(s):
    s = unicodedata.normalize("NFKC", str(s or "")).replace("‏", "").replace("‎", "")
    return re.sub(r"\s+", " ", s).strip()
recs = {}
def add(r, src):
    ref = clean(r.get("ref"))
    if not ref: return
    r = {k: clean(r.get(k)) for k in FIELDS if k != "source"}; r["ref"] = ref; r["source"] = src
    if ref in recs:
        for k in FIELDS:
            if not recs[ref][k] and r[k]: recs[ref][k] = r[k]
        recs[ref]["source"] += "+" + src if src not in recs[ref]["source"] else ""
    else: recs[ref] = r
counts = Counter()
if not base_only:
    for fn in sorted(glob.glob(ROOT + "/data/raw/page_*.json")):
        for t in json.load(open(fn))["data"]:
            add({"ref": t.get("referenceNumber"), "title": t.get("tenderName"), "agency": t.get("agencyName"),
                 "branch": t.get("branchName"), "activity": t.get("tenderActivityName"), "ttype": t.get("tenderTypeName"),
                 "publish_date": (t.get("submitionDate") or "")[:10]}, "crawl"); counts["crawl"] += 1
for t in json.load(open(W + "last-scan.json"))["tenders"]:
    ag = t.get("agency") or ""
    add({"ref": t["ref"], "title": t.get("title"), "agency": ag, "activity": t.get("activity"),
         "ttype": t.get("type"), "publish_date": t.get("publish_date")}, "last-scan"); counts["last-scan"] += 1
db = sqlite3.connect("file:" + W + "awards.db?mode=ro", uri=True)
for ref, name, agency, branch, act, tt, sub in db.execute("select ref,name,agency,branch,activity,ttype,submit_date from tenders"):
    add({"ref": ref, "title": name, "agency": agency, "branch": branch, "activity": act, "ttype": tt, "publish_date": (sub or "")[:10]}, "awards-db"); counts["awards-db"] += 1
wb = openpyxl.load_workbook(W + "Etimad-Awarded-Tenders.xlsx", read_only=True)
rows = wb["Awarded tenders"].iter_rows(values_only=True); h = next(rows); ix = {k: i for i, k in enumerate(h)}
for r in rows:
    add({"ref": r[ix["Ref"]], "title": r[ix["The request (tender)"]], "agency": r[ix["Agency"]], "branch": r[ix["Branch"]],
         "activity": r[ix["Industry / activity"]], "ttype": r[ix["Type"]], "publish_date": str(r[ix["Submitted"]] or "")[:10]}, "awarded-xlsx"); counts["awarded-xlsx"] += 1
rows = [r for r in recs.values() if r["title"] and r["activity"] and r["ttype"]]
out = os.path.join(ROOT, "data", "tenders_base.csv" if base_only else "tenders_all.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(sorted(rows, key=lambda r: r["ref"]))
print("raw rows per source:", dict(counts)); print("unique refs:", len(recs), "usable (title+activity+ttype):", len(rows)); print("->", out)
print("ttype:", Counter(r["ttype"] for r in rows).most_common())
print("activities:", len(set(r["activity"] for r in rows)))
for a, c in Counter(r["activity"] for r in rows).most_common(): print(f"{c:5d} {a}")
