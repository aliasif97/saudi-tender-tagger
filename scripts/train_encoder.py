#!/usr/bin/env python3
"""Fine-tune an Arabic BERT for one label. Usage: train_encoder.py <label: activity|ttype|sector> <hf-model> <out-dir> [epochs]"""
import json, os, sys, time, random
import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
sys.path.insert(0, os.path.dirname(__file__)); from common import *
label, mname, out = sys.argv[1], sys.argv[2], sys.argv[3]; EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 6; PATIENCE = int(sys.argv[5]) if len(sys.argv) > 5 else 2
COL = {"activity": "activity_label", "ttype": "ttype", "sector": "sector"}[label]
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tr = pd.read_csv(f"{ROOT}/data/published/train.csv", dtype=str).fillna(""); te = pd.read_csv(f"{ROOT}/data/published/test.csv", dtype=str).fillna("")
stats = json.load(open(f"{ROOT}/data/stats.json")); keep_t = {k for k, v in stats["ttype_counts"] if k != "other"}
for d in (tr, te): d["ttype"] = d.ttype.map(lambda x: x if x in keep_t else "other")
classes = sorted(tr[COL].unique()); c2i = {c: i for i, c in enumerate(classes)}
tr_, va = train_test_split(tr, test_size=0.1, random_state=SEED, stratify=tr[COL] if tr[COL].value_counts().min() >= 2 else None)
tok = AutoTokenizer.from_pretrained(mname)
model = AutoModelForSequenceClassification.from_pretrained(mname, num_labels=len(classes), id2label=dict(enumerate(classes)), label2id=c2i).to(dev)
def enc(df): return [make_input(t, a) for t, a in zip(df.title, df.agency)], [c2i.get(y, c2i.get("other", 0)) for y in df[COL]]
def batches(texts, ys, bs, shuffle):
    idx = list(range(len(texts)));
    if shuffle: random.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]; b = tok([texts[k] for k in j], truncation=True, max_length=128, padding=True, pad_to_multiple_of=32, return_tensors="pt")
        yield {k: v.to(dev) for k, v in b.items()}, torch.tensor([ys[k] for k in j]).to(dev)
def predict(texts, ys, top3=False):
    model.eval(); P = []; T = []
    with torch.no_grad():
        for b, _ in batches(texts, ys, 64, False):
            lg = model(**b).logits; P += lg.argmax(-1).cpu().tolist(); T += lg.topk(min(3, lg.shape[-1]), -1).indices.cpu().tolist()
    if dev == "mps": torch.mps.empty_cache()
    return (P, T) if top3 else P
Xt, yt = enc(tr_); Xv, yv = enc(va); Xe, ye = enc(te)
cnt = np.bincount(yt, minlength=len(classes)).astype(float); w = torch.tensor((cnt.max() / np.maximum(cnt, 1)) ** 0.5, dtype=torch.float).to(dev)
BS = 32; steps = EPOCHS * ((len(Xt) + BS - 1) // BS)
opt = torch.optim.AdamW(model.parameters(), lr=4e-5 if len(classes) < 10 else 5e-5, weight_decay=0.01)
sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps); lossf = torch.nn.CrossEntropyLoss(weight=w)
best, bad, t0, hist = -1, 0, time.time(), []; step_i = 0
os.makedirs(out, exist_ok=True)
for ep in range(EPOCHS):
    model.train(); tl = 0
    for b, y in batches(Xt, yt, BS, True):
        loss = lossf(model(**b).logits, y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True); tl += loss.item(); step_i += 1
        if dev == "mps" and step_i % 25 == 0: torch.mps.empty_cache()
    f1v = f1_score(yv, predict(Xv, yv), average="macro"); hist.append((ep + 1, round(tl, 2), round(f1v, 4)))
    print(f"epoch {ep+1} train_loss_sum={tl:.1f} val_macroF1={f1v:.4f} elapsed={time.time()-t0:.0f}s", flush=True)
    if f1v > best: best, bad = f1v, 0; model.save_pretrained(out); tok.save_pretrained(out)
    else:
        bad += 1
        if bad >= PATIENCE: print("early stop"); break
train_s = time.time() - t0
model = AutoModelForSequenceClassification.from_pretrained(out).to(dev)
yp, t3 = predict(Xe, ye, True); top3 = float(np.mean([y in t for y, t in zip(ye, t3)]))
res = {"label": label, "base_model": mname, "device": dev, "accuracy": accuracy_score(ye, yp), "macro_f1": f1_score(ye, yp, average="macro"),
       "top3_accuracy": top3, "best_val_macro_f1": best, "patience": PATIENCE, "max_epochs": EPOCHS, "epochs_run": len(hist), "history": hist, "train_seconds": round(train_s, 1), "n_classes": len(classes),
       "size_mb": round(sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) / 1e6, 1)}
json.dump(res, open(f"{ROOT}/outputs/encoder_{label}_metrics.json", "w"), indent=1)
pd.DataFrame({"ref": te.ref, "true": [classes[i] for i in ye], "pred": [classes[i] for i in yp]}).to_csv(f"{ROOT}/outputs/pred_encoder_{label}.csv", index=False)
print(res)
