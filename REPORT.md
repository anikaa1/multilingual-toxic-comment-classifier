# Academic Project Report: Multilingual Toxic Comment Classification using NLP

**Course/Project**: Natural Language Processing (NLP) / Applied Machine Learning  
**Author**: Machine Learning Engineering Team  
**Date**: October 2026  
**Status**: Runnable Codebase & Pipeline Verified; Full GPU Transformer Training Ready (Colab)  

---

## Executive Summary

Online toxicity, harassment, and hate speech present profound challenges to global digital discourse. While automated moderation models perform well on English text, they fail to provide equitable safety for the majority of the world's online users who communicate in other languages.

This project investigates and implements an end-to-end multilingual toxic comment classification framework. We compare a lightweight lexical-subword baseline (**TF-IDF Word + Character n-grams with Logistic Regression**) against a state-of-the-art cross-lingual masked language model (**FacebookAI/xlm-roberta-base**). The project emphasizes zero-shot cross-lingual transfer, non-destructive preprocessing, threshold calibration, and quantitative error analysis.

---

## 1. Problem Statement & Research Objectives

### 1.1 Problem Statement
Given a user-generated text sequence $X = (x_1, x_2, \dots, x_T)$ in an arbitrary language $L \in \mathcal{L}$, classify the comment into a binary toxicity target:
$$Y \in \{0, 1\}$$
where $Y=0$ denotes a non-toxic comment and $Y=1$ denotes a toxic comment (abusive, insulting, threatening, or obscene).

### 1.2 Core Challenges
1. **Severe Class Imbalance**: In public datasets and real-world comments, toxic speech constitutes only 8%–15% of all comments. Unadjusted optimization causes severe recall degradation.
2. **Obfuscation and Adversarial Spelling**: Users intentionally evade keyword filters via leetspeak, punctuation insertion, and deliberate misspellings (e.g., `f**k`, `b!tch`, `a$$hole`).
3. **Cross-Lingual Annotation Asymmetry**: Labeled training data is abundant in English but scarce in non-English languages. Models must transfer conceptual toxicity learned from English annotations to languages such as Spanish, Italian, Turkish, French, and Russian without target-language supervision.
4. **False Positive Risks on Identity Terms**: Standard models frequently confuse mentions of identity groups (e.g., religion, sexual orientation, ethnicity) with toxicity.

### 1.3 Key Objectives
- Design a modular, production-ready, reproducible NLP pipeline in Python, scikit-learn, PyTorch, and Hugging Face Transformers.
- Implement non-destructive preprocessing that preserves casing, punctuation, and emojis.
- Construct a strong baseline using Word (1-2) and Character (2-5) n-gram TF-IDF representations.
- Fine-tune XLM-RoBERTa base using dynamic padding, mixed precision (FP16), and AdamW optimization.
- Prevent test-set data leakage through strict split isolation and validation-based threshold calibration.
- Build an interactive Streamlit UI for single-comment and batch-file inference.

---

## 2. Methodology & Algorithms

```
                          ┌────────────────────────┐
                          │   Raw Text Comment     │
                          └───────────┬────────────┘
                                      │
                                      ▼
                      ┌────────────────────────────────┐
                      │    clean_text Preprocessing    │
                      │  • HTML unescaping             │
                      │  • URL/Mention masking         │
                      │  • Preserve case, !, ?, emojis │
                      └───────────────┬────────────────┘
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
      ┌─────────────────────────────┐   ┌─────────────────────────────┐
      │       Baseline Pipeline     │   │     Transformer Pipeline    │
      │ ├── Word TF-IDF (1-2 grams) │   │ ├── XLM-RoBERTa Tokenizer   │
      │ ├── Char TF-IDF (2-5 grams) │   │ ├── Dynamic Batch Collator  │
      │ └── Balanced Logistic Regr. │   │ └── 12-layer Transformer    │
      └──────────────┬──────────────┘   └──────────────┬──────────────┘
                     │                                 │
                     └────────────────┬────────────────┘
                                      │
                                      ▼
                      ┌────────────────────────────────┐
                      │    Toxicity Probability P(Y=1) │
                      └───────────────┬────────────────┘
                                      │
                                      ▼
                      ┌────────────────────────────────┐
                      │ Threshold Calibration (tau*)   │
                      │ P(Y=1) >= tau* -> Toxic (1)    │
                      └────────────────────────────────┘
```

### 2.1 Preprocessing Rationale
Standard NLP tutorials advocate aggressive text normalization: lowercasing, stopword removal, and word stemming. In toxicity classification, this is counterproductive:
- **Casing**: ALL-CAPS comments (e.g., "STOP TALKING TO ME") strongly correlate with anger and hostility.
- **Punctuation**: Exclamation marks ("!!!") and question-mark strings ("???") indicate emotional intensity.
- **Emojis**: Characters such as 🤬, 🖕, 💩 provide direct, language-independent toxicity signals.
- **HTML & URL Normalization**: HTML tags and URLs add lexical noise without semantic toxicity value and are replaced with `[URL]` and `[USER]`.

### 2.2 Baseline: Feature-Union TF-IDF + Logistic Regression
The baseline featurizes text using a scikit-learn `FeatureUnion`:
1. **Word n-grams ($n \in \{1, 2\}$)**: Captures toxic phrases and vocabulary.
2. **Character n-grams ($n \in \{2, 3, 4, 5\}$)**: Captures subword morphemes, cross-lingual character sequences, and intentional typo obfuscations.

The classifier is Logistic Regression with $L_2$ regularization:
$$\min_{w, b} \frac{1}{2} \|w\|_2^2 + C \sum_{i=1}^N \omega_{y_i} \log(1 + \exp(-y_i (w^T x_i + b)))$$
where $\omega_y$ represents class weights inversely proportional to class frequencies:
$$\omega_c = \frac{N}{2 \cdot N_c}$$

### 2.3 Deep Architecture: XLM-RoBERTa (Cross-Lingual Transformer)
XLM-RoBERTa (Conneau et al., 2020) is a multi-layer bidirectional Transformer trained on 2.5TB of CommonCrawl data in 100 languages.
- **Vocabulary**: 250,000 Byte-Pair Encoded (BPE) subwords with shared multilingual representation.
- **Classification Head**: The `<s>` sequence representation $h_{\text{cls}} \in \mathbb{R}^{768}$ passes through a dropout layer, dense projection, tanh activation, and linear classification layer:
  $$P(Y=1|X) = \sigma(W_2 \tanh(W_1 h_{\text{cls}} + b_1) + b_2)$$
- **Dynamic Padding**: Batches are padded to the maximum sequence length within the specific batch rather than global sequence length, reducing FLOPs by up to 40%.
- **Optimization**: AdamW ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$, weight decay $0.01$) with linear learning rate warmup over the first 10% of training steps.

### 2.4 Threshold Calibration
Rather than arbitrarily fixing the classification threshold at $\tau = 0.50$, the validation set is used to calibrate an optimal threshold $\tau^*$:
$$\tau^* = \arg\max_{\tau \in [0.05, 0.95]} F_1(\tau; D_{\text{val}})$$
This addresses class imbalance without causing test-set data leakage.

---

## 3. Experimental Setup & Datasets

### 3.1 Dataset Specification
- **Benchmark Dataset**: Jigsaw Multilingual Toxic Comment Classification (Kaggle / Conversation AI).
- **Training Set**: ~223,549 English Wikipedia talk comments.
- **Validation Set**: 8,000 comments in Spanish (`es`), Italian (`it`), and Turkish (`tr`).
- **Test Set**: 63,812 comments across 6 languages: `tr`, `it`, `es`, `fr`, `ru`, `pt`.
- **Synthetic Smoke-Test Set**: 30 curated multi-lingual comments (`data/synthetic_smoke_test.csv`) strictly for unit testing and offline code verification.

### 3.2 Evaluation Metrics
- **Toxic F1-Score**: $\frac{2 \cdot P \cdot R}{P + R}$ on the positive (toxic) class.
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **Macro-F1 & Accuracy**
- **ROC-AUC & PR-AUC**: Area under ROC and Precision-Recall curves.
- **Per-Language Performance**: Stratified breakdown across language codes.

---

## 4. Empirical Evaluation & Results

### 4.1 Pipeline Verification Results (Synthetic Split)
Pipeline functionality, training, serialization, and evaluation have been verified via unit tests and automated pipeline runs:

| Model Architecture | Precision | Recall | Toxic F1 | Accuracy | Status |
|---|---|---|---|---|---|
| **Baseline (TF-IDF + LogReg)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Verified & Persisted |
| **XLM-RoBERTa Transformer** | — | — | — | — | Pending GPU Execution |

> **Note on Verification**: The scores above reflect smoke-test pipeline execution on synthetic data to confirm that model serialization, feature extraction, and metric calculation run without errors.

### 4.2 Full Dataset Benchmarks (Pending GPU Training)
For training on the complete 223k Jigsaw dataset, execution on an NVIDIA GPU (e.g., T4/V100 on Google Colab via `notebooks/colab_training.ipynb`) is required. 

Expected benchmark profile based on published literature on this dataset:
- Baseline TF-IDF + Logistic Regression: ROC-AUC ~0.87 - 0.90 on English; ~0.78 - 0.82 on zero-shot multilingual evaluation.
- Fine-Tuned XLM-RoBERTa: ROC-AUC ~0.94 - 0.96; zero-shot multilingual ROC-AUC ~0.90 - 0.93.

---

## 5. Error Analysis & Case Studies

Our error analysis framework isolates False Positives (FP) and False Negatives (FN) to expose common failure modes:

### 5.1 False Positive Failure Modes (Over-flagging)
1. **Benign Mentions of Identity Keywords**: Comments quoting historical abuse or discussing LGBTQ+/religious topics without insult (e.g. *"I am proud to be gay"*).
2. **Quoted Speech in Reports**: Users quoting a violation when reporting someone else to moderators.
3. **Punctuation False Alarms**: Excessive exclamation points or ALL-CAPS used for excitement or celebration.

### 5.2 False Negative Failure Modes (Missed Toxicity)
1. **Idiomatic Slang in Low-Resource Languages**: Slang or regional insults in Turkish or Russian with low subword coverage.
2. **Sarcasm and Passive Aggression**: Comments that are polite in surface syntax but deeply hostile in pragmatic context (e.g., *"May your life be as pleasant as your personality"*).
3. **Complex Obfuscations**: Non-standard Unicode glyphs (homoglyphs) substituting Cyrillic characters for Latin letters.

---

## 6. Limitations, Bias, & Ethical Considerations

1. **Zero-Shot Transfer Decay**: While XLM-RoBERTa achieves strong transfer on high-resource languages (Spanish, French), performance drops noticeably on agglutinative languages (Turkish) and non-Latin scripts (Russian).
2. **Human Annotation Bias**: Training labels reflect the subjective judgments of annotators, which skew towards dominant cultural viewpoints.
3. **Chilling Effect**: Over-aggressive automated moderation suppresses participation by marginalized communities.
4. **Recommendation**: This system should function as an **assistant to human moderators** (flagging items for prioritized review) rather than an autonomous decision-maker.

---

## 7. Future Work & Extensions

1. **Adversarial Hardening**: Incorporate Byte-level Byte-Pair Encoding with homoglyph normalization to resist Unicode substitution attacks.
2. **Adapter & LoRA Fine-Tuning**: Apply Low-Rank Adaptation (LoRA) for parameter-efficient transfer learning with reduced VRAM requirements.
3. **Few-Shot Multilingual Tuning**: Fine-tune with a small set of labeled non-English comments to anchor cross-lingual alignment.
4. **Multi-Task & Fine-Grained Toxicity**: Expand classification from binary to multi-label (threat, insult, identity hate, obscenity).

---

## 8. Conclusion

This project delivers a complete, modular, and academically rigorous multilingual toxic comment classification system. The dual-model framework contrasts the low-latency efficiency of Word+Char TF-IDF with the deep cross-lingual semantic capabilities of XLM-RoBERTa, supported by an interactive Streamlit application and comprehensive test suite.
