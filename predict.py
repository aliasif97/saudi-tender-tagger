#!/usr/bin/env python3
"""Tag one Saudi government tender: Arabic title (+ agency) in, labels and confidences out.

  python predict.py --title "توريد أجهزة حاسب آلي محمول" --agency "وزارة الصحة"
  python predict.py --title "..." --agency "..." --json
  python predict.py --file tenders.csv        # columns: title, agency  -> prints JSON lines

Outputs: Etimad activity category (top 3), procurement type, sector (derived from the
predicted activity), kind supply/works/services (rules on activity, type and title).
Models are chosen per label in models/saudi-tender-tagger/config.json."""
import argparse, csv, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "scripts"))
from common import make_input, sector_of, kind_of

MODEL_DIR = os.path.join(HERE, "models", "saudi-tender-tagger")

class Head:
    def __init__(self, spec):
        self.kind, path = spec["type"], os.path.join(MODEL_DIR, spec["path"])
        if self.kind == "encoder":
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.torch = torch
            self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
            self.tok = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.dev).eval()
            self.labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
        else:
            import joblib
            self.wv, self.cv, self.clf = joblib.load(os.path.join(path, "model.joblib"))
            self.labels = list(self.clf.classes_)
    def probs(self, text):
        if self.kind == "encoder":
            b = self.tok([text], truncation=True, max_length=128, return_tensors="pt").to(self.dev)
            with self.torch.no_grad():
                return self.torch.softmax(self.model(**b).logits[0], -1).cpu().tolist()
        from scipy.sparse import hstack
        return self.clf.predict_proba(hstack([self.wv.transform([text]), self.cv.transform([text])]).tocsr())[0].tolist()
    def top(self, text, k=1):
        p = self.probs(text)
        return sorted(zip(self.labels, p), key=lambda x: -x[1])[:k]

def load_heads():
    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    return {k: Head(v) for k, v in cfg["heads"].items()}

def tag(heads, title, agency):
    text = make_input(title, agency)
    t0 = time.perf_counter()
    act = heads["activity"].top(text, 3); tt = heads["ttype"].top(text, 1)[0]
    activity = act[0][0]
    if "sector" in heads:
        sec = heads["sector"].top(text, 1)[0]; sector = {"label": sec[0], "confidence": round(sec[1], 4), "derived_from": "direct sector classifier"}
    else:
        sector = {"label": sector_of(activity), "derived_from": "predicted activity"}
    out = {
        "activity": {"label": activity, "confidence": round(act[0][1], 4),
                     "top3": [{"label": l, "confidence": round(p, 4)} for l, p in act]},
        "procurement_type": {"label": tt[0], "confidence": round(tt[1], 4)},
        "sector": sector,
        "kind": {"label": kind_of(activity, tt[0], title), "derived_from": "rules on activity, type and title"},
    }
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--title"); ap.add_argument("--agency", default=""); ap.add_argument("--file"); ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if not a.title and not a.file: ap.error("give --title or --file")
    t0 = time.perf_counter(); heads = load_heads(); load_s = time.perf_counter() - t0
    rows = list(csv.DictReader(open(a.file, encoding="utf-8"))) if a.file else [{"title": a.title, "agency": a.agency}]
    for r in rows:
        res = tag(heads, r["title"], r.get("agency", ""))
        if a.json or a.file: print(json.dumps({"title": r["title"], "agency": r.get("agency", ""), **res}, ensure_ascii=False))
        else:
            print(f"Title:    {r['title']}\nAgency:   {r.get('agency','')}")
            print("Top-3 activities:")
            for i, x in enumerate(res["activity"]["top3"], 1): print(f"  {i}. {x['label']}  p={x['confidence']:.3f}")
            print(f"Type:     {res['procurement_type']['label']}  ({res['procurement_type']['confidence']:.2f})")
            sc = res["sector"].get("confidence"); print(f"Sector:   {res['sector']['label']}" + (f"  p={sc:.3f}" if sc else "") + f"   Kind: {res['kind']['label']}")
            print(f"Latency:  {res['latency_ms']} ms per tender (model load {load_s:.1f} s)")
if __name__ == "__main__": main()
