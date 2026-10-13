import os, re, json, unicodedata
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SEED = 42
def norm_text(s):
    s = unicodedata.normalize("NFKC", str(s or "")).replace("‏", "").replace("‎", "")
    return re.sub(r"\s+", " ", s).strip()
def make_input(title, agency):
    return f"{norm_text(title)} | {norm_text(agency)}"
def load_json(p): return json.load(open(os.path.join(ROOT, p), encoding="utf-8"))


# ---- rule-based derived labels (shared by make_splits.py and predict.py) ----
_TR = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"})
def okey(a): return re.sub(r"\s+", " ", str(a).translate(_TR)).strip()
_SM = _KM = None
def _maps():
    global _SM, _KM
    if _SM is None:
        _SM = {okey(k): v for k, v in load_json("labels/activity_to_sector.json")["map"].items()}
        _KM = {okey(k): v for k, v in load_json("labels/activity_to_kind.json")["map"].items()}
    return _SM, _KM
def sector_of(activity):
    return _maps()[0].get(okey(activity), "other")
def kind_of(activity, ttype, title):
    k = _maps()[1].get(okey(activity))
    if k is None:
        k = "works" if activity.startswith("مقاولات") else "supply" if activity.startswith(("تجارة", "صُنع", "صنع", "صناعة", "البيع بالتجزئة", "انتاج", "إنتاج")) else "services"
    t = norm_text(title)
    if k == "mixed":
        if any(w in t for w in ("توريد", "شراء", "تأمين", "تامين", "supply")): return "supply"
        if any(w in t for w in ("إنشاء", "انشاء", "تنفيذ", "تشييد", "ترميم", "مشروع")): return "works"
        return "services"
    if ttype == "شراء مباشر" and k == "services" and any(w in t for w in ("توريد", "شراء", "supply", "purchase")): return "supply"
    return k
