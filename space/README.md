---
title: Saudi Tender Tagger
emoji: 🏷️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Tag Arabic Saudi government tender titles
---

# Saudi Tender Tagger

Paste an Arabic Etimad tender title (and optionally the issuing agency). The demo returns the Etimad activity category (top 3 with confidence), the procurement type, the sector and the kind (supply, works or services).

الصق عنوان منافسة حكومية سعودية من منصة اعتماد (والجهة اختياريا) لتحصل على النشاط (أعلى ثلاثة احتمالات) ونوع المنافسة والقطاع ونوع الأعمال.

- Model: https://huggingface.co/chaliasif/saudi-tender-tagger
- Dataset: https://huggingface.co/datasets/chaliasif/etimad-tender-titles

If the model repo is private, add a read-access `HF_TOKEN` secret in the Space settings.
