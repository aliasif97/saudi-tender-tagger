#!/usr/bin/env python3
"""tenders_all.csv -> normalized labels -> stratified 85/15 split (seed 42) -> data/published/*"""
import csv, json, os, sys
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(__file__))
from common import *
MIN_N = 8
df = pd.read_csv(f"{ROOT}/data/tenders_all.csv", dtype=str).fillna("")
df["ref"] = df["ref"].astype(str)
# 1. Etimad sometimes publishes "<parent activity> - <sub activity>" (crawl rows). The primary label is the parent,
#    so rows from every source share one label set; the sub activity is kept as activity_detail.
import re
df["activity_detail"] = df.activity
df["activity"] = df.activity.map(lambda a: re.split(r"\s*-\s*", a, maxsplit=1)[0].strip())
# fold spelling variants (hamza / ya / ta marbuta) onto the most frequent surface form
def okey(a): return re.sub(r"\s+", " ", a.translate(str.maketrans({"أ":"ا","إ":"ا","آ":"ا","ى":"ي","ة":"ه"}))).strip()
_c = Counter(df.activity); _best = {}
for a, n in _c.most_common(): _best.setdefault(okey(a), a)
df["activity"] = df.activity.map(lambda a: _best[okey(a)])
df["text"] = [make_input(t, a) for t, a in zip(df.title, df.agency)]
# 2. collapse rare classes (counted over the whole dataset) into "other"
def collapse(col):
    c = Counter(df[col]); rare = {k for k, v in c.items() if v < MIN_N}
    return df[col].map(lambda x: "other" if x in rare else x), len(rare), sum(c[k] for k in rare)
df["activity_label"], n_rare_a, n_rows_a = collapse("activity")
df["ttype_label"], n_rare_t, n_rows_t = collapse("ttype")
# 3. derived labels
amap = load_json("labels/activity_to_sector.json"); smap = amap["map"]
kmap = load_json("labels/activity_to_kind.json")
smap_k = {okey(k): v for k, v in smap.items()}
kmap_k = {okey(k): v for k, v in kmap["map"].items()}
unmapped = sorted({a for a in df.activity if okey(a) not in smap_k})
def sector(a): return smap_k.get(okey(a), "other")
df["sector"] = df.activity.map(sector)
SUPPLY_WORDS = ("توريد", "شراء", "supply", "purchase")
def kind(a, tt, title):
    k = kmap_k.get(okey(a))
    if k is None:
        k = "works" if a.startswith("مقاولات") else "supply" if a.startswith(("تجارة", "صُنع", "صنع", "صناعة", "البيع بالتجزئة", "انتاج", "إنتاج")) else "services"
    if k == "mixed":   # activity does not decide; use title verbs
        t = norm_text(title)
        if any(w in t for w in ("توريد", "شراء", "تأمين", "تامين", "supply")): return "supply"
        if any(w in t for w in ("إنشاء", "انشاء", "تنفيذ", "تشييد", "ترميم", "مشروع")): return "works"
        return "services"
    if tt == "شراء مباشر" and k == "services" and any(w in norm_text(title) for w in SUPPLY_WORDS): return "supply"
    return k
df["kind"] = [kind(a, t, ti) for a, t, ti in zip(df.activity, df.ttype, df.title)]
train, test = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df.activity_label)
os.makedirs(f"{ROOT}/data/published", exist_ok=True)
cols = ["ref", "title", "agency", "branch", "activity", "activity_detail", "activity_label", "ttype", "sector", "kind", "publish_date", "source"]
for name, part in (("train", train), ("test", test)):
    part = part.sort_values("ref")
    part[cols].to_csv(f"{ROOT}/data/published/{name}.csv", index=False, encoding="utf-8")
    with open(f"{ROOT}/data/published/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in part[cols].to_dict("records"): f.write(json.dumps(r, ensure_ascii=False) + "\n")
tk = set(zip(train.title.map(norm_text), train.agency.map(norm_text)))
dup = sum((norm_text(t), norm_text(a)) in tk for t, a in zip(test.title, test.agency))
stats = {"total": len(df), "train": len(train), "test": len(test),
         "activity_classes_raw": df.activity.nunique(), "activity_classes_after_collapse": df.activity_label.nunique(),
         "rare_activities_collapsed": n_rare_a, "rows_in_collapsed_activities": n_rows_a,
         "ttype_classes_raw": df.ttype.nunique(), "ttype_rare_collapsed": n_rare_t, "ttype_rows_collapsed": n_rows_t,
         "test_rows_with_exact_title_agency_in_train": dup,
         "unmapped_activities_to_other": unmapped, "sector_counts": Counter(df.sector).most_common(), "kind_counts": Counter(df.kind).most_common(),
         "ttype_counts": Counter(df.ttype_label).most_common(), "source_counts": Counter(df.source).most_common()}
json.dump(stats, open(f"{ROOT}/data/stats.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in stats.items() if k not in ("source_counts",)}, ensure_ascii=False, indent=1))
