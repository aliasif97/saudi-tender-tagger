#!/usr/bin/env python3
"""TF-IDF (word 1-2 + char_wb 2-5) + logistic regression, one model per label. Usage: train_baseline.py"""
import json, os, sys, time, joblib
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(__file__)); from common import *
tr = pd.read_csv(f"{ROOT}/data/published/train.csv", dtype=str).fillna(""); te = pd.read_csv(f"{ROOT}/data/published/test.csv", dtype=str).fillna("")
ttl = lambda d: [make_input(t, a) for t, a in zip(d.title, d.agency)]
def ttype_label(d, ref):  # same collapse as activity: map unseen/rare to other using train counts
    return d.ttype
LABELS = {"activity": "activity_label", "ttype": "ttype", "sector": "sector", "kind": "kind"}
# collapse rare ttype exactly as in make_splits (<8 overall -> other); recompute from stats
stats = json.load(open(f"{ROOT}/data/stats.json")); keep_t = {k for k, v in stats["ttype_counts"] if k != "other"}
tr["ttype"] = tr.ttype.map(lambda x: x if x in keep_t else "other"); te["ttype"] = te.ttype.map(lambda x: x if x in keep_t else "other")
res = {}
os.makedirs(f"{ROOT}/outputs", exist_ok=True)
for name, col in LABELS.items():
    t0 = time.time()
    a, b = train_test_split(tr, test_size=0.1, random_state=SEED, stratify=tr[col] if tr[col].value_counts().min() >= 2 else None)
    def fit(df, C):
        wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
        X = hstack([wv.fit_transform(ttl(df)), cv.fit_transform(ttl(df))]).tocsr()
        m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(X, df[col]); return wv, cv, m
    def pred(p, df): wv, cv, m = p; return m.predict(hstack([wv.transform(ttl(df)), cv.transform(ttl(df))]).tocsr())
    best = max([1, 10, 50], key=lambda C: f1_score(b[col], pred(fit(a, C), b), average="macro"))
    p = fit(tr, best); yp = pred(p, te)
    res[name] = {"C": best, "accuracy": accuracy_score(te[col], yp), "macro_f1": f1_score(te[col], yp, average="macro"),
                 "seconds": round(time.time() - t0, 1), "n_classes": int(tr[col].nunique())}
    pd.DataFrame({"ref": te.ref, "true": te[col], "pred": yp}).to_csv(f"{ROOT}/outputs/pred_baseline_{name}.csv", index=False)
    if name in ("activity", "ttype", "sector"):
        os.makedirs(f"{ROOT}/models/saudi-tender-tagger/{name}_tfidf", exist_ok=True)
        joblib.dump(p, f"{ROOT}/models/saudi-tender-tagger/{name}_tfidf/model.joblib", compress=3)
    print(name, res[name], flush=True)
json.dump(res, open(f"{ROOT}/outputs/baseline_metrics.json", "w"), indent=1)
