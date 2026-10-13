---
language:
- ar
license: mit
inference: false
pipeline_tag: text-classification
base_model: CAMeL-Lab/bert-base-arabic-camelbert-mix
datasets:
- chaliasif/etimad-tender-titles
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

**Try it locally:** `python predict.py --title "..." --agency "..."` (a hosted demo is planned).

Reads the Arabic title of a Saudi government tender (and the issuing agency) and predicts the Etimad activity category and the procurement type. It also gives a coarse sector and a supply / works / services label, both derived from the predicted activity.

**ملخص بالعربية:** نموذج يقرأ عنوان المنافسة الحكومية السعودية واسم الجهة، ويتوقع نشاط اعتماد ونوع المنافسة، ثم يشتق القطاع ونوع الأعمال (توريد / أعمال / خدمات). الهدف مساعدة المنشآت الصغيرة والمتوسطة على فرز المنافسات بسرعة. الدقة على اختبار منفصل: النشاط 61.1% (54 فئة) وصحة وجود النشاط الصحيح ضمن أفضل ثلاثة اقتراحات 80.4%، نوع المنافسة 87.9%. التصنيفات الحقيقية في اعتماد يضعها كل جهة بنفسها وفيها تداخل، لذلك لا يصلح النموذج كمرجع نهائي. للمراجعة البشرية فقط.

## Use

```bash
python predict.py --title "توريد وتركيب أجهزة حاسب آلي محمول" --agency "وزارة الصحة"
python predict.py --file tenders.csv --json     # columns: title, agency
```
Output: top-3 activities with probabilities, procurement type, sector, kind, latency. About 0.26 s per tender on Apple MPS after a 1.6 s warm-up call, 3 to 5 s model load. Weights sit in a subfolder, so the Hub inference widget is off; use `predict.py` or the demo Space.

Models per label (`models/saudi-tender-tagger/config.json`):

| Label | Shipped model | Why |
|---|---|---|
| activity (54 classes) | `activity_encoder`: CAMeLBERT-mix (Apache-2.0) fine-tuned, 437 MB | macro-F1 0.434 vs 0.354 for the baseline; top-3 0.804 vs 0.799 |
| procurement type (5 values) | `ttype_tfidf`: TF-IDF + logistic regression, 3.3 MB | higher macro-F1 than the encoder |
| sector (19) | derived from predicted activity with `labels/activity_to_sector.json` | macro-F1 0.607 vs 0.535 for a direct TF-IDF sector classifier |
| kind (3) | rules on activity, type and title | |

## Data
7,016 public tender listings from tenders.etimad.sa (visitor pages): an unfiltered newest-first crawl on 13 Oct 2026 (6,207 rows, published 21 Jul to 13 Oct 2026) plus 914 earlier locally collected tenders (mostly awarded tenders from 2019 to 2020 and a recent scan), de-duplicated by reference number. Only tender fields are published: no bidders, winners, award values, emails or phones. Details in `DATASET-CARD.md`. Input text is `title | agency`, max 128 tokens.

Split: stratified 85/15 on the activity label, seed 42. Train 5,963, test 1,053. Activities with fewer than 8 rows (58 of 111, covering 135 rows) are merged into `other`, leaving 54 activity classes. A 10% slice of train is the validation set for early stopping and model selection.

## Results (untouched test set, 1,053 tenders)

| Label | TF-IDF + LR | CAMeLBERT-mix (shipped) | MARBERTv2, 6 epochs (not shipped) |
|---|---|---|---|
| activity, accuracy | 0.613 | 0.611 | 0.617 |
| activity, macro-F1 | 0.354 | **0.434** | 0.434 |
| activity, top-3 accuracy | 0.799 | **0.804** | 0.810 |
| procurement type, accuracy | 0.879 (shipped) | not trained | 0.882 |
| procurement type, macro-F1 | **0.832** (shipped) | not trained | 0.819 |
| sector from predicted activity, accuracy | 0.657 | 0.690 | 0.699 |
| sector from predicted activity, macro-F1 | 0.492 | **0.607** | 0.612 |
| kind from predicted activity and type, accuracy / macro-F1 | 0.893 / 0.868 | 0.882 / 0.860 | 0.891 / 0.872 |

Shipped: CAMeLBERT activity encoder, TF-IDF procurement-type model, sector derived from the shipped activity prediction, kind by rules.

**Sector routes.** A direct TF-IDF sector classifier (19 classes, trained on sector as its own label) scores 0.684 accuracy / 0.535 macro-F1. Deriving sector from the shipped activity prediction scores 0.690 / 0.607, so the derived route ships. A direct encoder sector head was started and abandoned for time and was never evaluated.

**Top-3.** For screening, the correct activity is among the three suggestions for 80.4% of tenders (shipped model), against 61.1% for the single top suggestion.

**Training of the shipped activity encoder.** Planned for up to 14 epochs with early stopping on validation macro-F1 (patience 3). The run was stopped by the time cap after 5 epochs (about 57 minutes wall clock; the machine stalled for long stretches, so epochs ranged from 2.5 to 20 minutes). The best validation checkpoint, epoch 5 (validation macro-F1 0.4045), was kept and evaluated. It had not plateaued, so a longer run would probably score higher. The learning-rate schedule was set for 14 epochs, so it had not decayed when the run stopped. Baselines train in 7 s to 2 minutes on CPU.

**Not shipped, measured.** MARBERTv2 for 6 epochs: 0.617 accuracy, 0.434 macro-F1, 0.810 top-3 (876 s). It is not shipped because the base model has no stated license (see License). A MARBERTv2 procurement-type encoder (4 epochs, 578 s) scored 0.882 / 0.819 and lost to TF-IDF on macro-F1, so the type model is TF-IDF and no CAMeLBERT type model was trained.

Other checks: on the 1,030 test rows whose exact title and agency are not in train, activity accuracy is 0.603 (shipped encoder) and 0.604 (baseline), so leakage from repeated titles is small. Macro-F1 on procurement type differs by 0.013 between models, which is within noise.

What goes wrong:
- Activity: the confusions are between neighbouring labels that agencies use interchangeably. Building contracting vs general construction (14 and 8 test rows), spare parts vs equipment trade (9), telecom vs IT (7 each way), equipment trade vs building and electrical materials (7). Education, specialised construction and petroleum scored F1 of zero because each has only 5 to 8 test rows.
- Procurement type: direct purchase vs general competition is confused in both directions (about 115 errors out of 1,053). The title says little about the agency's choice of method.
- Sector accuracy is bounded by activity noise, since sector is a many-to-one regrouping of activity.
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
python scripts/train_encoder.py activity CAMeL-Lab/bert-base-arabic-camelbert-mix models/saudi-tender-tagger/activity_encoder 14 3
# if stopped early: python scripts/eval_checkpoint.py activity models/saudi-tender-tagger/activity_encoder <epochs> <best_val_f1> <seconds> 14 3
python scripts/extra_top3.py   # baseline top-3
python scripts/evaluate.py
```
(`build_dataset.py` reads the author's private local sources; the published CSV/JSONL in `data/published/` is enough to retrain.)

## License
Code and the fine-tuned weights: MIT, except that the weights derive from the base model below. Dataset: CC BY 4.0 over our selection and labels; the underlying listings are public government information (see `LICENSE`).

**Base-model license check (Hub API, 13 Oct 2026):** `CAMeL-Lab/bert-base-arabic-camelbert-mix` states `apache-2.0`. `UBC-NLP/MARBERTv2` and `aubmindlab/bert-base-arabertv2` state no license in their model cards. Because an unstated license gives no clear right to redistribute or use commercially, the shipped model is built on CAMeLBERT-mix. The MARBERTv2 run (same macro-F1, 0.434) is reported above for reference only and its weights are not distributed. Apache-2.0 requires keeping the notice and attribution for CAMeLBERT (Inoue et al., 2021, CAMeL Lab, NYU Abu Dhabi).
