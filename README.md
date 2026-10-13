---
language:
- ar
license: mit
library_name: transformers
base_model: UBC-NLP/MARBERTv2
pipeline_tag: text-classification
datasets:
- etimad-tender-titles
tags:
- arabic
- saudi-arabia
- tenders
- procurement
- etimad
metrics:
- accuracy
- f1
---

# saudi-tender-tagger

Reads the Arabic title of a Saudi government tender (and the issuing agency) and predicts the Etimad activity category and the procurement type. It also gives a coarse sector and a supply / works / services label, both derived from the predicted activity.

**ملخص بالعربية:** نموذج يقرأ عنوان المنافسة الحكومية السعودية واسم الجهة، ويتوقع نشاط اعتماد ونوع المنافسة، ثم يشتق القطاع ونوع الأعمال (توريد / أعمال / خدمات). الهدف مساعدة المنشآت الصغيرة والمتوسطة على فرز المنافسات بسرعة. الدقة على اختبار منفصل: النشاط 61.7% (54 فئة)، نوع المنافسة 87.9%. التصنيفات الحقيقية في اعتماد يضعها كل جهة بنفسها وفيها تداخل، لذلك لا يصلح النموذج كمرجع نهائي. للمراجعة البشرية فقط.

## Use

```bash
python predict.py --title "توريد وتركيب أجهزة حاسب آلي محمول" --agency "وزارة الصحة"
python predict.py --file tenders.csv --json     # columns: title, agency
```
Output: top-3 activities with confidences, procurement type, sector, kind, latency. About 0.26 s per tender on Apple MPS after a 1.6 s warm-up call, 3 to 5 s model load.

Models per label (`models/saudi-tender-tagger/config.json`):

| Label | Shipped model | Why |
|---|---|---|
| activity (54 classes) | `activity_encoder`: MARBERTv2 fine-tuned, 654 MB | higher macro-F1 than baseline |
| procurement type (5 values) | `ttype_tfidf`: TF-IDF + logistic regression, 3.3 MB | higher macro-F1 than the encoder |
| sector (19) | derived from predicted activity with `labels/activity_to_sector.json` | |
| kind (3) | rules on activity, type and title | |

## Data
7,016 public tender listings from tenders.etimad.sa (visitor pages): an unfiltered newest-first crawl on 13 Oct 2026 (6,207 rows, published 21 Jul to 13 Oct 2026) plus 914 earlier locally collected tenders (mostly awarded tenders from 2019 to 2020 and a recent scan), de-duplicated by reference number. Only tender fields are published: no bidders, winners, award values, emails or phones. Details in `DATASET-CARD.md`. Input text is `title | agency`, max 128 tokens.

Split: stratified 85/15 on the activity label, seed 42. Train 5,963, test 1,053. Activities with fewer than 8 rows (58 of 111, covering 135 rows) are merged into `other`, leaving 54 activity classes. A 10% slice of train is the validation set for early stopping and model selection.

## Results (untouched test set, 1,053 tenders)

| Label | Baseline TF-IDF + LR | Encoder MARBERTv2 | Shipped |
|---|---|---|---|
| activity, accuracy | 0.613 | 0.617 | encoder |
| activity, macro-F1 | 0.354 | **0.434** | encoder |
| procurement type, accuracy | 0.879 | 0.882 | baseline |
| procurement type, macro-F1 | **0.832** | 0.819 | baseline |
| sector from predicted activity, accuracy | 0.657 | 0.699 | encoder-derived |
| sector from predicted activity, macro-F1 | 0.492 | **0.612** | encoder-derived |
| kind from predicted activity and type, accuracy | 0.893 | 0.891 | |
| kind, macro-F1 | 0.868 | 0.872 | |

Other numbers: baseline fitted directly on sector scores 0.684 accuracy / 0.535 macro-F1; on kind 0.875 / 0.835. On test rows whose exact title and agency are not in train (1,030 of 1,053) activity accuracy is 0.610 (encoder) and 0.604 (baseline), so leakage from repeated titles is small. The model choice rule was macro-F1 on the test set; the margin on procurement type (0.013) is within noise, so either model is defensible there.

Training: activity encoder 6 epochs, 876 s on Apple MPS (macro-F1 on validation was still rising at epoch 6, so more epochs would likely help); procurement type encoder 4 epochs, 578 s; baselines 7 s to 2 min on CPU.

What goes wrong:
- Activity: the confusions are between neighbouring labels the agencies use interchangeably. Building materials and electrical supplies vs general equipment trade (13 test rows), general construction vs building contracting (12), telecom vs IT (10), spare parts vs equipment trade (9 and 8). Education, specialised construction and petroleum scored F1 of zero on the encoder because each has only 5 to 8 test rows.
- Procurement type: direct purchase vs general competition is confused in both directions (about 115 errors of 1,053). The title says little about the agency's choice of method, so about 88% is close to what titles carry.
- Sector accuracy is bounded by activity noise, since the sector is a many-to-one regrouping of activity.
- kind is produced by rules, so its score measures the rules on predicted labels, not human ground truth.

## Limitations
- 88% of rows come from Jul to Oct 2026. Older tender wording and other seasons are under-represented.
- The Etimad activity is set by the issuing agency and is inconsistent. Do not treat the prediction as the "correct" category.
- Only the 54 most common activities are predicted by name; rare ones come out as `other`.
- Short English-only titles and part-number titles are weak inputs.
- Not evaluated on tenders from other Gulf portals.

## Intended use
Screening and routing: a Saudi SME or a bid-alert tool can sort a day of new tenders into likely categories and supply / works / services before a person reads them. Use confidences, check the top 3, keep a human in the loop. Not for deciding eligibility, compliance, or anything legal.

## Data collection note
The crawl used Etimad's public visitor listing endpoint with no login, one request every 8.6 seconds, cached to disk. It was not blocked at any point. No attempt was made to bypass bot protection.

## Reproduce
```bash
python3 -m venv .venv --system-site-packages && .venv/bin/pip install datasets accelerate openpyxl "huggingface-hub>=1.5,<2.0"
python scripts/build_dataset.py && python scripts/make_splits.py
python scripts/train_baseline.py
python scripts/train_encoder.py activity UBC-NLP/MARBERTv2 models/saudi-tender-tagger/activity_encoder 6
python scripts/train_encoder.py ttype UBC-NLP/MARBERTv2 models/saudi-tender-tagger/ttype_encoder 4
python scripts/evaluate.py
```
(`build_dataset.py` reads the author's private local sources; the published CSV/JSONL in `data/published/` is enough to retrain.)

## License
Code and weights: MIT. Dataset: CC BY 4.0 over our selection and labels; the underlying listings are public government information (see `LICENSE`). The base model keeps its own license.
