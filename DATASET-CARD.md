---
license: cc-by-4.0
language:
- ar
- en
task_categories:
- text-classification
pretty_name: Etimad tender titles (Arabic) with activity, type, sector and kind labels
size_categories:
- 1K<n<10K
tags:
- arabic
- saudi-arabia
- government-procurement
- tenders
---

# Etimad tender titles

Dataset card. Arabic summary at the end.

## What it is
7,016 Saudi government tender listings (title, issuing agency, branch) with Etimad's own activity category and procurement type, plus two derived labels. Built for the model `saudi-tender-tagger`.

## Provenance
All rows are public listings from the visitor pages of Etimad (tenders.etimad.sa). Nothing is behind a login. Four local sources, merged by tender reference number:

| Source | Rows | Notes |
|---|---|---|
| Unfiltered newest-first listing crawl | 6,207 | Collected 13 Oct 2026, publish dates 21 Jul 2026 to 13 Oct 2026, 260 pages of 24, one request per 8.6 s |
| Earlier tender watch DB (awarded tenders) | 334 + 475 | Publish dates 2019 to 2020 (awarded tenders only) |
| Tender watch scan (aviation and defence buyers) | 105 | Overlaps the crawl |

Before merging there were 914 unique tenders from the three local sources (the "base" set). The crawl grew it to 7,030 unique refs; 7,016 have a title, activity and type and are kept. 88% of rows are from Jul to Oct 2026, so the set reflects recent listings, and the 2019 to 2020 rows are awarded tenders (almost all open competitions).

Not included: bidder names, winner names, award values, emails, phones, websites. A regex scan of all text fields found no emails, URLs or Saudi phone numbers.

## Fields
`ref` Etimad reference number; `title` tender title as published (Arabic, about 17% contain Latin words); `agency`; `branch`; `activity` parent Etimad activity (gold label as published); `activity_detail` the sub-activity string when Etimad published one ("parent - sub"); `activity_label` the training label (activities with fewer than 8 rows in the full set are merged into `other`: 58 activities, 135 rows); `ttype` procurement type; `sector`; `kind`; `publish_date`; `source`.

## Labels
- **activity**: chosen by the issuing agency when it posts the tender. 111 parent activities, 54 classes after merging rare ones. It is noisy: the same item (fire extinguishers, plumbing parts, diesel fuel, bottled water) appears under several activities.
- **ttype**: 5 values. Direct purchase 3,536; general competition 3,378; framework agreement 89; two-stage 12; income-sharing 1 (the single row is shown as `other` in model evaluation).
- **sector**: 19 coarse classes from a hand-written table, `labels/activity_to_sector.json`. Ambiguous activities are listed there with a note. Treat it as a convenience grouping, not an official taxonomy.
- **kind**: supply 3,821, services 2,279, works 916. Rule-derived from activity, type and title verbs (rules in `labels/activity_to_kind.json` and `scripts/common.py`). It is not human annotated, so scores on it measure the rules, not ground truth.

## Splits
Stratified on `activity_label`, seed 42: train 5,963, test 1,053 (15%). 23 test rows have a title and agency identical to a train row.

## Limitations
Recency skew (above). Source skew: the older rows exist only because they were awarded, so they over-represent open competitions. Agency labelling noise caps any model. Titles from some agencies are truncated by the portal.

## License
CC BY 4.0 for the rows we assembled and labelled. The source text is public government listing data that we do not own; see LICENSE. Remove rows on request from the portal operator.

## Updates
Weekly refresh adds newly listed tenders under `updates/` (one file per date); `train` and `test` never change. Each tag freezes the repository at that date.

| Date | New rows | Total rows | Tag |
|---|---|---|---|
| 2026-10-13 | 119 | 7,135 | v2026.10.13 |

## ملخص بالعربية
مجموعة بيانات من 7,016 منافسة حكومية سعودية منشورة علنا على منصة اعتماد (الصفحات العامة للزوار)، تتضمن العنوان والجهة والنشاط حسب تصنيف اعتماد ونوع المنافسة، مع تصنيفين مشتقين (القطاع ونوع الأعمال: توريد أو أعمال أو خدمات). لا تتضمن أسماء المتقدمين أو الفائزين أو قيم الترسية أو بيانات اتصال. الترخيص CC BY 4.0 للتجميع والتصنيف، والنص الأصلي معلومات حكومية عامة.
