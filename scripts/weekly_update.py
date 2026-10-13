#!/usr/bin/env python3
"""Weekly refresh of the Etimad tender-titles dataset.

  python scripts/weekly_update.py [--dry-run] [--max-pages 40] [--no-upload]

1. known refs  <- data/tenders_all.csv
2. crawl the unfiltered newest-first visitor listing exactly like scripts/crawl.py (same endpoint,
   headers, >= 8.6 s between requests, sequential) until two consecutive pages add no new ref, or --max-pages
3. normalise like build_dataset.py / make_splits.py (parent activity, spelling fold, frozen v1 label set)
4. append to data/tenders_all.csv, write data/published/updates/<date>.jsonl + .csv (train/test never touched)
5. refresh the "## Updates" table in DATASET-CARD.md
6. upload updates/ + card (as README.md) to the dataset repo and tag v<YYYY.MM.DD>   (skipped with --dry-run)
Idempotent: refs already in tenders_all.csv are never added twice."""
import argparse, csv, json, os, re, sys, time, datetime, logging
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from common import norm_text, sector_of, kind_of, okey

REPO = "chaliasif/etimad-tender-titles"
BASE = "https://tenders.etimad.sa/Tender/AllSupplierTendersForVisitorAsync"
H = {"X-Requested-With": "XMLHttpRequest", "Referer": "https://tenders.etimad.sa/Tender/AllTendersForVisitor"}
MIN_GAP = max(8.6, float(os.environ.get("MIN_GAP", "8.6")))
ALL_FIELDS = ["ref", "title", "agency", "branch", "activity", "ttype", "publish_date", "source"]
PUB_FIELDS = ["ref", "title", "agency", "branch", "activity", "activity_detail", "activity_label", "ttype", "sector", "kind", "publish_date", "source"]
ALL_CSV = os.path.join(ROOT, "data", "tenders_all.csv")
UPD_DIR = os.path.join(ROOT, "data", "published", "updates")
CARD = os.path.join(ROOT, "DATASET-CARD.md")
log = logging.getLogger("weekly")

def setup_log():
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s")
    log.setLevel(logging.INFO)
    for h in (logging.FileHandler(os.path.join(ROOT, "logs", "weekly_update.log"), encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt); log.addHandler(h)

def frozen_labels():
    """v1 class list + surface forms, read from the published splits (never recomputed from counts)."""
    import pandas as pd
    df = pd.concat([pd.read_csv(os.path.join(ROOT, "data", "published", f"{n}.csv"), dtype=str).fillna("") for n in ("train", "test")])
    classes = set(df.activity_label) - {"other"}
    surf = {}
    for a, _ in Counter(df.activity).most_common(): surf.setdefault(okey(a), a)
    for a in classes: surf.setdefault(okey(a), a)
    return classes, surf

def normalise(t, classes, surf, src):
    g = lambda k: norm_text(t.get(k))
    full = g("tenderActivityName")
    parent = re.split(r"\s*-\s*", full, maxsplit=1)[0].strip()
    parent = surf.get(okey(parent), parent)
    row = {"ref": g("referenceNumber"), "title": g("tenderName"), "agency": g("agencyName"), "branch": g("branchName"),
           "activity": parent, "activity_detail": full, "activity_label": parent if parent in classes else "other",
           "ttype": g("tenderTypeName"), "sector": sector_of(parent), "publish_date": (t.get("submitionDate") or "")[:10], "source": src}
    row["kind"] = kind_of(parent, row["ttype"], row["title"])
    return row

def fetch(pg, state):
    from curl_cffi import requests
    qs = ("MultipleSearch=&TenderCategory=&ReferenceNumber=&TenderNumber=&AgencyCode=&PublishDateId=&TenderTypeId="
          "&TenderActivityId=&SortDirection=DESC&Sort=SubmitionDate&PageSize=24&PageNumber=%d&IsSearch=false" % pg)
    for attempt in range(3):
        wait = MIN_GAP - (time.time() - state["last"])
        if wait > 0: time.sleep(wait)
        state["last"] = time.time()
        try:
            j = requests.get(BASE + "?" + qs, impersonate="safari", timeout=45, headers=H).json()
            return j.get("data") or []
        except Exception as e:
            log.info(f"page {pg} attempt {attempt+1} failed: {str(e)[:80]}"); time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"BLOCKED at page {pg}: 3 failures, stopping without retrying around it")

def update_card(date, new_total, total, tag):
    txt = open(CARD, encoding="utf-8").read()
    hdr = "| Date | New rows | Total rows | Tag |\n|---|---|---|---|\n"
    m = re.search(r"## Updates\n(.*?)(?=\n## )", txt, re.S)
    rows = re.findall(r"^\| (\d{4}-\d\d-\d\d) \| ([\d,]+) \| ([\d,]+) \| (\S+) \|$", m.group(1), re.M) if m else []
    if not rows: rows = [("2026-10-13", "0", f"{total - new_total:,}", "v1.0")]
    rows = [r for r in rows if r[0] != date] + [(date, f"{new_total:,}", f"{total:,}", tag)]
    block = ("## Updates\nWeekly refresh adds newly listed tenders under `updates/` (one file per date); `train` and `test` never change. "
             "Each tag freezes the repository at that date.\n\n" + hdr + "".join(f"| {a} | {b} | {c} | {d} |\n" for a, b, c, d in rows) + "\n")
    if m: txt = txt[:m.start()] + block.rstrip("\n") + "\n" + txt[m.end():]
    else: txt = txt.replace("## ملخص بالعربية", block + "## ملخص بالعربية", 1)
    open(CARD, "w", encoding="utf-8").write(txt)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--max-pages", type=int, default=40)
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args(); setup_log()
    today = datetime.date.today().isoformat(); src = f"weekly-{today}"
    log.info(f"start {today} dry_run={a.dry_run} max_pages={a.max_pages}")
    with open(ALL_CSV, encoding="utf-8", newline="") as f: known = {r["ref"] for r in csv.DictReader(f)}
    classes, surf = frozen_labels()
    log.info(f"known refs {len(known)}, frozen classes {len(classes)} (+other)")
    new, seen, state, empty_streak, pages = [], set(), {"last": 0.0}, 0, 0
    try:
        for pg in range(1, a.max_pages + 1):
            data = fetch(pg, state); pages += 1
            if not data: log.info(f"page {pg}: empty, end of listing"); break
            added = 0
            for t in data:
                r = normalise(t, classes, surf, src)
                if not (r["ref"] and r["title"] and r["activity"] and r["ttype"]) or r["ref"] in known or r["ref"] in seen: continue
                seen.add(r["ref"]); new.append(r); added += 1
            log.info(f"page {pg}: {len(data)} rows, {added} new (total new {len(new)})")
            empty_streak = 0 if added else empty_streak + 1
            if empty_streak >= 2: log.info("two consecutive pages with no new refs, stopping"); break
    except RuntimeError as e:
        log.info(str(e)); return 2
    log.info(f"crawled {pages} pages, {len(new)} new rows")
    if a.dry_run:
        log.info("dry run: nothing written"); [log.info("  " + json.dumps(r, ensure_ascii=False)) for r in new[:3]]; return 0
    if new:
        with open(ALL_CSV, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, ALL_FIELDS, extrasaction="ignore"); w.writerows(new)
        os.makedirs(UPD_DIR, exist_ok=True)
        jp, cp = os.path.join(UPD_DIR, today + ".jsonl"), os.path.join(UPD_DIR, today + ".csv")
        with open(jp, "a", encoding="utf-8") as f:
            for r in new: f.write(json.dumps({k: r[k] for k in PUB_FIELDS}, ensure_ascii=False) + "\n")
        rows = [json.loads(l) for l in open(jp, encoding="utf-8")]
        with open(cp, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, PUB_FIELDS); w.writeheader(); w.writerows(rows)
    jp = os.path.join(UPD_DIR, today + ".jsonl")
    day_new = sum(1 for _ in open(jp, encoding="utf-8")) if os.path.exists(jp) else 0
    with open(ALL_CSV, encoding="utf-8", newline="") as f: total = sum(1 for _ in csv.DictReader(f))
    tag = "v" + today.replace("-", ".")
    update_card(today, day_new, total, tag)
    log.info(f"day new rows {day_new}, total {total}, card updated, tag {tag}")
    if a.no_upload: return 0
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(path_or_fileobj=CARD, path_in_repo="README.md", repo_id=REPO, repo_type="dataset", commit_message=f"Card: update table {today}")
    if os.path.isdir(UPD_DIR) and os.listdir(UPD_DIR):
        api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=UPD_DIR, path_in_repo="updates", commit_message=f"Weekly update {today}: {day_new} new rows")
    api.create_tag(REPO, tag=tag, repo_type="dataset", tag_message=f"Weekly refresh {today}: +{day_new}, total {total}", exist_ok=True)
    log.info(f"uploaded and tagged {tag}")
    return 0
if __name__ == "__main__": sys.exit(main())
