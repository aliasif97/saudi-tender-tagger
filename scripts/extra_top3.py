#!/usr/bin/env python3
"""Top-3 accuracy on the test set for the TF-IDF baseline and for a saved encoder dir. Usage: extra_top3.py [encoder_dir label]"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from common import *
tr = pd.read_csv(f"{ROOT}/data/published/train.csv", dtype=str).fillna(""); te = pd.read_csv(f"{ROOT}/data/published/test.csv", dtype=str).fillna("")
ttl = lambda d: [make_input(t, a) for t, a in zip(d.title, d.agency)]
res = json.load(open(f"{ROOT}/outputs/top3.json")) if os.path.exists(f"{ROOT}/outputs/top3.json") else {}
if len(sys.argv) == 1:
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True); cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    X = hstack([wv.fit_transform(ttl(tr)), cv.fit_transform(ttl(tr))]).tocsr()
    m = LogisticRegression(C=50, max_iter=2000).fit(X, tr.activity_label)
    P = m.predict_proba(hstack([wv.transform(ttl(te)), cv.transform(ttl(te))]).tocsr())
    top = np.array(m.classes_)[np.argsort(-P, 1)[:, :3]]
    res["activity_baseline"] = float(np.mean([y in t for y, t in zip(te.activity_label, top)]))
else:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    d = sys.argv[1]; tok = AutoTokenizer.from_pretrained(d); mod = AutoModelForSequenceClassification.from_pretrained(d).eval()
    X = ttl(te); hits = []
    for i in range(0, len(X), 64):
        b = tok(X[i:i+64], truncation=True, max_length=128, padding=True, return_tensors="pt")
        with torch.no_grad(): t = mod(**b).logits.topk(3, -1).indices.tolist()
        hits += [te.activity_label.iloc[i + j] in [mod.config.id2label[k] for k in tt] for j, tt in enumerate(t)]
    res[sys.argv[2]] = float(np.mean(hits))
json.dump(res, open(f"{ROOT}/outputs/top3.json", "w"), indent=1); print(res)
