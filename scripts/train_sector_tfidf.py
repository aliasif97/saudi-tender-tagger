#!/usr/bin/env python3
"""Fit and save the direct TF-IDF + LR sector head (C=50, as chosen in train_baseline.py)."""
import os, sys, joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(__file__)); from common import *
tr = pd.read_csv(f"{ROOT}/data/published/train.csv", dtype=str).fillna("")
ttl = [make_input(t, a) for t, a in zip(tr.title, tr.agency)]
wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True); cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
X = hstack([wv.fit_transform(ttl), cv.fit_transform(ttl)]).tocsr()
m = LogisticRegression(C=50, max_iter=2000).fit(X, tr.sector)
os.makedirs(f"{ROOT}/models/saudi-tender-tagger/sector_tfidf", exist_ok=True)
joblib.dump((wv, cv, m), f"{ROOT}/models/saudi-tender-tagger/sector_tfidf/model.joblib", compress=3); print("saved")
