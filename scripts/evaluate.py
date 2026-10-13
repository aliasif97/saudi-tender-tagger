#!/usr/bin/env python3
"""Compare baseline vs encoder on the untouched test split. Writes outputs/metrics.json and outputs/eval_report.txt
and models/saudi-tender-tagger/config.json (which model ships per label)."""
import json, os, sys
from collections import Counter
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
sys.path.insert(0, os.path.dirname(__file__)); from common import *
O = f"{ROOT}/outputs"; te = pd.read_csv(f"{ROOT}/data/published/test.csv", dtype=str).fillna(""); tr = pd.read_csv(f"{ROOT}/data/published/train.csv", dtype=str).fillna("")
te = te.set_index("ref")
seen = set(zip(tr.title.map(norm_text), tr.agency.map(norm_text)))
novel = [(norm_text(t), norm_text(a)) not in seen for t, a in zip(te.title, te.agency)]
te["novel"] = novel
def m(y, p): return {"accuracy": round(accuracy_score(y, p), 4), "macro_f1": round(f1_score(y, p, average="macro"), 4)}
out, lines, ship = {}, [], {}
P = {}
for lab in ("activity", "ttype"):
    for model in ("baseline", "encoder"):
        P[(lab, model)] = pd.read_csv(f"{O}/pred_{model}_{lab}.csv", dtype=str).set_index("ref")
for lab in ("activity", "ttype"):
    rows = {}
    for model in ("baseline", "encoder"):
        d = P[(lab, model)].loc[te.index]
        r = m(d.true, d.pred); r["accuracy_novel_titles"] = round(accuracy_score(d.true[te.novel], d.pred[te.novel]), 4)
        rows[model] = r
        c = Counter(zip(d.true[d.true != d.pred], d.pred[d.true != d.pred])).most_common(6)
        lines.append(f"\n[{lab} / {model}] {r}\n  top confusions (true -> predicted, count):")
        lines += [f"    {a} -> {b}: {n}" for (a, b), n in c]
        f1s = pd.Series(f1_score(d.true, d.pred, average=None, labels=sorted(set(d.true)), zero_division=0), index=sorted(set(d.true)))
        sup = d.true.value_counts()
        w = f1s[sup[f1s.index] >= 5].sort_values().head(5)
        lines.append("  weakest classes (support>=5): " + "; ".join(f"{k} F1={v:.2f} n={sup[k]}" for k, v in w.items()))
    win = max(rows, key=lambda k: (rows[k]["macro_f1"], rows[k]["accuracy"])); rows["ships"] = win; out[lab] = rows; ship[lab] = win
# derived labels from predicted activity vs truth
for model in ("baseline", "encoder"):
    d = P[("activity", model)].loc[te.index]
    ps = [sector_of(x) for x in d.pred]; pk = [kind_of(x, tt, ti) for x, tt, ti in zip(d.pred, P[("ttype", model)].loc[te.index].pred, te.title)]
    out.setdefault("sector_derived", {})[model] = m(te.sector, ps); out.setdefault("kind_derived", {})[model] = m(te.kind, pk)
    lines.append(f"\n[sector derived from predicted activity / {model}] {out['sector_derived'][model]}")
    lines.append(f"[kind rules on predicted activity+type / {model}] {out['kind_derived'][model]}")
    c = Counter((a, b) for a, b in zip(te.sector, ps) if a != b).most_common(5)
    lines.append("  top sector confusions: " + "; ".join(f"{a}->{b}:{n}" for (a, b), n in c))
bm = json.load(open(f"{O}/baseline_metrics.json"))
out["baseline_direct_sector_and_kind"] = {k: bm[k] for k in ("sector", "kind")}
out["test_n"] = len(te); out["test_novel_title_n"] = int(te.novel.sum())
json.dump(out, open(f"{O}/metrics.json", "w"), indent=1, ensure_ascii=False)
open(f"{O}/eval_report.txt", "w").write(json.dumps(out, indent=1, ensure_ascii=False) + "\n" + "\n".join(lines))
cfg = {"name": "saudi-tender-tagger", "input": "title + ' | ' + agency", "max_length": 128,
       "heads": {l: ({"type": "encoder", "path": f"{l}_encoder"} if ship[l] == "encoder" else {"type": "tfidf", "path": f"{l}_tfidf"}) for l in ship},
       "derived": {"sector": "labels/activity_to_sector.json", "kind": "labels/activity_to_kind.json + rules in scripts/common.py"}}
json.dump(cfg, open(f"{ROOT}/models/saudi-tender-tagger/config.json", "w"), indent=1)
print(open(f"{O}/eval_report.txt").read())
