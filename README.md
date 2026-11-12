# Multilingual Toxic Comment Classifier using NLP

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

A complete, production-grade NLP system for **multilingual toxic comment classification**, comparing **TF-IDF + Logistic Regression (Word & Character n-grams)** against **Fine-Tuned XLM-RoBERTa (Cross-Lingual Deep Transformer)**.

---

## Table of Contents
1. [Project Overview & Architecture](#project-overview--architecture)
2. [Folder Structure](#folder-structure)
3. [Environment Setup & Installation](#environment-setup--installation)
4. [Dataset Acquisition & Structure](#dataset-acquisition--structure)
5. [Model Training](#model-training)
6. [Evaluation & Benchmarking](#evaluation--benchmarking)
7. [Inference (CLI & Streamlit App)](#inference-cli--streamlit-app)
8. [Testing & Quality Assurance](#testing--quality-assurance)
9. [Metrics Explanation & Evaluation Rationale](#metrics-explanation--evaluation-rationale)
10. [Limitations, Bias, & Cross-Lingual Nuances](#limitations-bias--cross-lingual-nuances)
11. [Troubleshooting & FAQ](#troubleshooting--faq)

---

## 1. Project Overview & Architecture

Online discussions across platforms (Wikipedia, social media, community forums) are plagued by hostile, abusive, and toxic comments. While English moderation systems are abundant, moderating non-English or multilingual discussions remains a severe challenge due to annotation scarcity.

This project addresses **Multilingual Toxic Comment Classification** through two complementary paradigms:
1. **Lightweight Baseline**: Feature-union TF-IDF with **Word n-grams (1-2)** and **Character n-grams (2-5)** paired with a **Logistic Regression classifier with balanced class weighting**. The character n-grams capture cross-lingual subword roots, intentional typos, and obfuscations (e.g. `sh!t`, `a$$hole`).
2. **Deep Pretrained Transformer**: Fine-tuning **`FacebookAI/xlm-roberta-base`** (Conneau et al., 2020), utilizing dynamic padding, mixed-precision (FP16), AdamW optimization, linear learning rate warmup, and validation-based early stopping. XLM-RoBERTa leverages a 250,000 subword vocabulary pre-trained across 100 languages to achieve **zero-shot cross-lingual transfer**.

### Architectural Pipeline
```
Raw Multilingual Comment
         │
         ▼
Text Preprocessing Pipeline (clean_text)
 ├── Strip HTML entities & tags
 ├── Mask URLs -> [URL]
 ├── Mask user mentions -> [USER]
 ├── Remove non-printable control characters
 └── Preserve true casing, punctuation (!, ?), and emojis (🤬, 🖕)
         │
         ├─────────────────────────────────────────┐
         ▼                                         ▼
Baseline Pipeline                         Transformer Pipeline
 ├── Word TF-IDF (1-2 n-grams)             ├── XLM-RoBERTa Tokenizer
 ├── Char TF-IDF (2-5 n-grams)             ├── PyTorch Dynamic Batch Collator
 └── Logistic Regression                   └── 12-layer XLM-RoBERTa Classifier
         │                                         │
         └────────────────────┬────────────────────┘
                              ▼
                  Toxicity Probability [0, 1]
                              ▼
                Threshold Calibration (0.05-0.95)
                              ▼
                     Binary Verdict:
               Non-Toxic (0) vs. Toxic (1)
```

---

## 2. Folder Structure

```
Toxic comment classifier/
├── api/
│   ├── __init__.py            # API package initialization
│   └── main.py                # FastAPI server: model loading, endpoints, static mounts
├── static/
│   ├── index.html             # Apple-inspired web dashboard (Classify & Results pages)
│   ├── style.css              # Minimalist typography, layout, and color styling
│   └── app.js                 # Routing, fetch prediction, canvas chart rendering
├── app.py                     # Streamlit interactive web application
├── train.py                   # Unified CLI training script
├── evaluate.py                # Model evaluation and comparison script
├── predict.py                 # Single-text and batch-file inference CLI
├── preprocess.py              # Text cleaning, normalization, and stratified splitting
├── config.py                  # Centralized configuration dataclasses & hardware detection
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation
├── REPORT.md                  # Academic report (methodology, results, ethics)
├── .gitignore                 # Ignore caches, virtual environments, large checkpoints
├── data/
│   ├── README.md              # Dataset schema and documentation
│   ├── download_instructions.md # Guide to download Kaggle Jigsaw dataset
│   └── synthetic_smoke_test.csv # Small synthetic dataset for smoke tests
├── models/
│   └── .gitkeep               # Saved model weights (baseline.joblib, xlm-roberta/)
├── notebooks/
│   ├── exploration.ipynb      # In-depth exploratory data analysis
│   └── colab_training.ipynb   # Google Colab GPU training notebook
├── src/
│   ├── __init__.py            # Package exports
│   ├── dataset.py             # PyTorch Dataset and dynamic batch loader
│   ├── baseline.py            # TF-IDF + Logistic Regression pipeline
│   ├── transformer_model.py   # XLM-RoBERTa fine-tuning and inference
│   └── metrics.py             # Academic evaluation metrics, plots, error cases
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py  # Unit tests for text normalization & leakage
    ├── test_metrics.py        # Unit tests for evaluation metrics calculation
    └── test_prediction.py     # Unit tests for model fitting and inference
```

---

## 3. Environment Setup & Installation

### Option A: Local Python Environment
```bash
# 1. Clone / navigate to workspace
cd "Toxic comment classifier"

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Google Colab GPU Training (Recommended for XLM-RoBERTa)
For deep transformer training, use `notebooks/colab_training.ipynb` with a free T4 GPU.

---

## 4. Dataset Acquisition & Structure

The primary supported dataset is the **Jigsaw Multilingual Toxic Comment Classification** dataset (Kaggle).

1. Follow instructions in [data/download_instructions.md](file:///Users/anikakhajuria/Downloads/Toxic%20comment%20classifier/data/download_instructions.md).
2. Place the extracted files into the `data/` directory:
   - `data/jigsaw-toxic-comment-train.csv` (English training data, ~223k rows)
   - `data/validation.csv` (Multilingual validation data: Italian, Spanish, Turkish)
   - `data/test.csv` (Multilingual test data: 6 languages)

### Offline Smoke-Testing
If you do not have the full dataset downloaded yet, the codebase includes a lightweight multi-lingual synthetic dataset at `data/synthetic_smoke_test.csv` allowing you to immediately run and test the complete pipeline using the `--use-synthetic` flag.

---

## 5. Model Training

### Training Baseline (TF-IDF + Logistic Regression)
```bash
# Quick smoke-test on synthetic data:
python train.py --model baseline --use-synthetic

# Full training on downloaded Jigsaw dataset:
python train.py --model baseline --train-path data/jigsaw-toxic-comment-train.csv
```
The trained baseline model will be saved to `models/baseline_model.joblib`.

### Fine-Tuning XLM-RoBERTa Transformer
```bash
# Quick smoke-test (1 epoch, small batch):
python train.py --model transformer --use-synthetic --epochs 1 --batch-size 8

# Full training (recommended on GPU / Colab):
python train.py \
    --model transformer \
    --train-path data/jigsaw-toxic-comment-train.csv \
    --val-path data/validation.csv \
    --epochs 3 \
    --batch-size 16 \
    --lr 2e-5 \
    --max-length 128
```
The best checkpoint and tokenizer are saved to `models/xlm-roberta-toxic/`.

---

## 6. Evaluation & Benchmarking

To evaluate your trained model(s) on held-out test data:
```bash
# Evaluate on synthetic test split:
python evaluate.py --model all --use-synthetic

# Evaluate on full multilingual test data:
python evaluate.py --model all --test-path data/test.csv --threshold 0.50
```

### Outputs Generated:
- `results/baseline_tf-idf_metrics.json`: Accuracy, Precision, Recall, Toxic F1, ROC-AUC, PR-AUC, Confusion Matrix.
- `results/baseline_tf-idf_confusion_matrix.png`: Heatmap of True Positives, False Positives, False Negatives, and True Negatives.
- `results/baseline_tf-idf_roc_curve.png`: ROC Curve.
- `results/baseline_tf-idf_pr_curve.png`: Precision-Recall Curve.
- `results/models_comparison_summary.csv`: Side-by-side comparison table.

---

## 7. Inference (CLI & Streamlit App)

### Single Comment Inference (CLI)
```bash
python predict.py --model baseline --text "You are a complete idiot, get out of here!"
python predict.py --model baseline --text "Merci beaucoup pour cet article très informatif!"
```

### Batch File Inference (CLI)
```bash
python predict.py \
    --model baseline \
    --file data/synthetic_smoke_test.csv \
    --output results/predictions.csv \
    --threshold 0.50
```

### Web Dashboard (FastAPI + Modern UI)
Launch the restrained, Apple-inspired web dashboard:
```bash
./venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
```
Open `http://127.0.0.1:8000` in your browser.

Features:
- **Classify**: Centred text analysis with one-click multilingual examples (English, Spanish, Turkish, and polite comments). Side-by-side probability and verdict cards for Baseline and XLM-RoBERTa.
- **Results**: Live test set F1 headline, per-language performance canvas bar chart, collapsible full metric comparison tables (Test and Validation), confusion matrices, and batch CSV classification.

### Streamlit Web Application (Alternative)
Launch the legacy Streamlit UI:
```bash
streamlit run app.py
```
**App Features:**
- **Interactive Comment Analysis**: Real-time toxicity probability bar, threshold slider, text normalization display, and multilingual quick-select examples (English, Spanish, French, German, Italian, Turkish).
- **Batch CSV Classification**: Drag-and-drop CSV scoring with live progress and CSV export.
- **Evaluation Viewer**: Visualizes confusion matrices, ROC/PR curves, and comparison metrics.
- **Demo Fallback Transparency**: Explicit badge alerting users if model weights are missing from disk.

---

## 8. Testing & Quality Assurance

Run the automated test suite:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
Or with pytest:
```bash
pytest tests/ -v
```

---

## 9. Metrics Explanation & Evaluation Rationale

In toxicity detection, standard Accuracy is deeply deceptive due to severe class imbalance (often >90% non-toxic comments). A dummy model predicting "Non-toxic" for every comment achieves >90% accuracy while failing completely at its primary purpose.

We evaluate:
1. **Toxic F1-Score ($F_1$)**: The harmonic mean of Precision and Recall for the toxic class ($1$).
2. **Precision**: $\frac{TP}{TP + FP}$ — The proportion of flagged comments that are genuinely toxic. High precision prevents unjust censorship of innocent users.
3. **Recall**: $\frac{TP}{TP + FN}$ — The proportion of actual toxic comments successfully flagged. High recall prevents toxic abuse from slipping through unnoticed.
4. **ROC-AUC & PR-AUC**: Threshold-independent metrics assessing ranking quality across all operating points.
5. **Threshold Calibration**: Scans decision thresholds $\tau \in [0.05, 0.95]$ on the validation set to optimize F1 according to operational priorities.

---

## 10. Limitations, Bias, & Cross-Lingual Nuances

1. **Zero-Shot Cross-Lingual Transfer**: Models fine-tuned purely on English comments transfer well to romance and germanic languages due to shared subword representations, but exhibit higher false-negative rates on idiomatic slang or non-Latin scripts (e.g. Russian, Arabic).
2. **False Positives on Identity Terms**: Lexical bias occurs when benign mentions of identity terms (e.g. *gay*, *lesbian*, *muslim*, *jewish*, *black*, *immigrant*) trigger false alarms due to statistical co-occurrence with abuse in training data.
3. **Sarcasm and Masked Toxicity**: Subtle aggression, passive-aggressive remarks, and sarcasm frequently slip past automated models.
4. **Ethical Recommendation**: Classifiers should be deployed as **assistive tools for human moderators**, not autonomous decision engines.

---

## 11. Troubleshooting & FAQ

- **`ModuleNotFoundError: No module named 'torch'`**: Ensure your virtual environment is active (`source venv/bin/activate`) and run `pip install -r requirements.txt`.
- **`OutOfMemoryError` on GPU**: In `train.py`, decrease `--batch-size` to `8` or `4`, and use `--max-length 128` instead of `256`.
- **Baseline model not found in app**: Run `python train.py --model baseline --use-synthetic` to generate initial baseline weights.
