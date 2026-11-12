# Dataset Documentation: Multilingual Toxic Comment Classification

## 1. Primary Dataset: Jigsaw Multilingual Toxic Comment Classification

This project is built around the **Jigsaw Multilingual Toxic Comment Classification** dataset (hosted on Kaggle by Jigsaw and Google's Conversation AI team).

### Dataset Overview
- **Source**: [Kaggle Competition - Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification)
- **License**: Creative Commons Attribution 4.0 International (CC-BY 4.0) / Public Domain
- **Domain**: Wikipedia talk page comments and Civil Comments platform discussions.

### Critical Academic Context: Zero-Shot Cross-Lingual Transfer
A defining challenge of this dataset is that **the primary high-volume training set is in English (`jigsaw-toxic-comment-train.csv`)**, while the **validation and test sets are multilingual**:
- **Training Data**: ~223,549 comments in English (`en`) labelled for toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). In binary classification, `toxic >= 0.5` or `toxic == 1` is used as positive class.
- **Validation Data**: ~8,000 human-annotated comments across Italian (`it`), Spanish (`es`), and Turkish (`tr`).
- **Test Data**: ~63,812 human-annotated comments across 6 languages:
  - Turkish (`tr`)
  - Italian (`it`)
  - Spanish (`es`)
  - French (`fr`)
  - Russian (`ru`)
  - Portuguese (`pt`)

> **Academic Rigor Note**: Because training data is predominantly English, multilingual models (such as XLM-RoBERTa) rely on **cross-lingual zero-shot transfer**—learning semantic representations of toxicity from English annotations and projecting them into shared multilingual subword embedding space. We document per-language evaluation metrics explicitly to avoid overclaiming language capabilities.

---

## 2. File Format & Column Mapping

The pipeline supports configurable column names via `config.py` or CLI arguments:

| Raw Dataset Column | Internal Standard | Description |
|---|---|---|
| `comment_text` or `text` | `text` | The text of the comment |
| `toxic` or `label` | `label` | Binary target: 0 (non-toxic), 1 (toxic) |
| `lang` | `lang` | Two-letter ISO language code (`en`, `es`, `fr`, `it`, `tr`, `ru`, `pt`) |
| `id` | `id` | Unique sample identifier |

---

## 3. Synthetic Smoke Test Dataset

For local debugging, continuous integration, and pipeline verification without downloading gigabytes of competition data, this repository includes:
- `data/synthetic_smoke_test.csv`

> **NOTICE**: The synthetic dataset is strictly intended for **smoke testing the code pipeline, unit tests, and verifying environment dependencies**. It contains a minimal set of curated non-toxic and toxic samples across multiple languages. **Do not use the synthetic dataset to evaluate or report model performance.**

---

## 4. Downloading the Full Dataset

See [download_instructions.md](file:///Users/anikakhajuria/Downloads/Toxic%20comment%20classifier/data/download_instructions.md) for full commands and options.
