"""Saudi Tender Tagger demo (Gradio). Arabic tender title (+ agency) in, labels out."""
import os, sys, json
import gradio as gr

REPO = "chaliasif/saudi-tender-tagger"

def _model_root():
    local = os.environ.get("TAGGER_LOCAL_DIR")
    if local: return os.path.abspath(local)
    from huggingface_hub import snapshot_download
    return snapshot_download(REPO, token=os.environ.get("HF_TOKEN"))

ROOT = _model_root()
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import predict
# predict.py resolves models/saudi-tender-tagger next to itself; on the Hub the weights sit at the repo root.
_nested = os.path.join(ROOT, "models", "saudi-tender-tagger")
predict.MODEL_DIR = _nested if os.path.isfile(os.path.join(_nested, "config.json")) else ROOT
HEADS = predict.load_heads()

EXAMPLES = [
    ["مشروع دعم وصيانة أنظمة حماية البيانات والحوكمة السيبرانية بجامعة حائل", "جامعة حائل"],
    ["رخصة الدعم الفني لنظام صرف الأدوية الآلي", "مركز الأمير سلطان لأمراض وجراحة القلب"],
    ["الصيانة الطارئة لمصدات درء اخطار السيول (بلدية محافظة ابوعريش)", "بلدية محافظة ابى عريش"],
    ["تأمين قطع غيار و إصلاح اليات التابعة لمديرية المنطقة الشرقية والإدارات التابعة لها و الفروع DM 26 62 0143", "ادارة الشؤون الفنية المركزية بالمنطقة الشرقية"],
    ["توريد مستلزمات السفن جلالة الملك (لمركز التموين البحري بالاسطول الشرقي )", "القوات البحرية"],
]

def _lab(v):
    return v.get("label", "") if isinstance(v, dict) else str(v)

def _conf(v):
    c = v.get("confidence") if isinstance(v, dict) else None
    return f"{c:.1%}" if isinstance(c, (int, float)) else "-"

def run(title, agency):
    if not (title or "").strip():
        return [], [], "{}"
    res = predict.tag(HEADS, title.strip(), (agency or "").strip())
    act = res.get("activity")
    top = (act.get("top3") if isinstance(act, dict) else None) or ([act] if act else [])
    act_rows = [[i + 1, _lab(x), _conf(x)] for i, x in enumerate(top)]
    other = []
    for key, name in (("procurement_type", "Procurement type / نوع المنافسة"), ("sector", "Sector / القطاع"), ("kind", "Kind / نوع الأعمال")):
        if key in res:
            v = res[key]; other.append([name, _lab(v), _conf(v)])
    known = {"activity", "procurement_type", "sector", "kind"}
    for k, v in res.items():
        if k not in known and not isinstance(v, (dict, list)):
            other.append([k, str(v), "-"])
    return act_rows, other, json.dumps(res, ensure_ascii=False, indent=2)

CSS = ".rtl textarea, .rtl input {direction: rtl; text-align: right;}"
with gr.Blocks(title="Saudi Tender Tagger") as demo:
    gr.Markdown("# Saudi Tender Tagger / مصنّف المنافسات السعودية\n"
                "Paste an Arabic Etimad tender title; get the activity category, procurement type, sector and kind. "
                "الصق عنوان المنافسة بالعربية لتحصل على النشاط ونوع المنافسة والقطاع ونوع الأعمال.")
    title = gr.Textbox(label="Tender title / عنوان المنافسة", lines=3, elem_classes="rtl", rtl=True)
    agency = gr.Textbox(label="Agency (optional) / الجهة (اختياري)", elem_classes="rtl", rtl=True)
    btn = gr.Button("Tag / صنّف", variant="primary")
    act = gr.Dataframe(headers=["#", "Activity / النشاط", "Confidence / الثقة"], label="Activity, top 3 / النشاط", interactive=False, wrap=True)
    oth = gr.Dataframe(headers=["Field / الحقل", "Label / التصنيف", "Confidence / الثقة"], label="Other labels / تصنيفات أخرى", interactive=False, wrap=True)
    raw = gr.Code(label="Raw output (JSON)", language="json")
    btn.click(run, [title, agency], [act, oth, raw])
    title.submit(run, [title, agency], [act, oth, raw])
    gr.Examples(EXAMPLES, [title, agency], label="Examples / أمثلة")
    gr.Markdown("Model: [chaliasif/saudi-tender-tagger](https://huggingface.co/chaliasif/saudi-tender-tagger) | "
                "Data: [chaliasif/etimad-tender-titles](https://huggingface.co/datasets/chaliasif/etimad-tender-titles)")

if __name__ == "__main__":
    demo.launch(css=CSS, server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
