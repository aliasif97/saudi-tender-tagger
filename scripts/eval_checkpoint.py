#!/usr/bin/env python3
"""Evaluate the best saved checkpoint of a run that was stopped early by the time cap.
Usage: eval_checkpoint.py <label> <model_dir> <epochs_run> <best_val_macro_f1> <train_seconds_wall> <max_epochs> <patience>"""
import json, os, sys
import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, os.path.dirname(__file__)); from common import *
label, d, ep, bv, secs, mx, pat = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
te = pd.read_csv(f"{ROOT}/data/published/test.csv", dtype=str).fillna("")
tok = AutoTokenizer.from_pretrained(d); mod = AutoModelForSequenceClassification.from_pretrained(d).eval()
dev = "mps" if torch.backends.mps.is_available() else "cpu"; mod.to(dev)
X = [make_input(t, a) for t, a in zip(te.title, te.agency)]; P, T = [], []
for i in range(0, len(X), 64):
    b = tok(X[i:i+64], truncation=True, max_length=128, padding=True, return_tensors="pt").to(dev)
    with torch.no_grad(): lg = mod(**b).logits
    P += lg.argmax(-1).tolist(); T += lg.topk(3, -1).indices.tolist()
id2 = mod.config.id2label; pred = [id2[i] for i in P]; true = list(te.activity_label if label == "activity" else te[label])
top3 = float(np.mean([y in [id2[k] for k in t] for y, t in zip(true, T)]))
res = {"label": label, "base_model": json.load(open(f"{d}/config.json")).get("_name_or_path", ""), "device": dev, "accuracy": accuracy_score(true, pred),
       "macro_f1": f1_score(true, pred, average="macro"), "top3_accuracy": top3, "best_val_macro_f1": bv, "epochs_run": ep, "max_epochs": mx,
       "patience": pat, "train_seconds": secs, "n_classes": len(id2), "stopped_by": "time cap",
       "size_mb": round(sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) / 1e6, 1)}
json.dump(res, open(f"{ROOT}/outputs/encoder_{label}_metrics.json", "w"), indent=1)
pd.DataFrame({"ref": te.ref, "true": true, "pred": pred}).to_csv(f"{ROOT}/outputs/pred_encoder_{label}.csv", index=False); print(res)
