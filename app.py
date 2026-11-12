"""Streamlit Web Application: Multilingual Toxic Comment Classifier.

Features:
- Single comment classification with real-time probability meter and threshold slider.
- Batch CSV upload and export.
- Pre-loaded multilingual example comments.
- Model benchmark metrics and visualization viewer.
- Explicit status badge distinguishing trained models from demo fallback.
"""

import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st

# Add workspace to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import APP_CONFIG, BASELINE_CONFIG, TRANSFORMER_CONFIG, RESULTS_DIR, get_default_device
from preprocess import clean_text
from src.baseline import BaselineToxicityClassifier

# Set page configuration
st.set_page_config(
    page_title="Multilingual Toxic Comment Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for polished academic presentation
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #dee2e6;
        margin-bottom: 12px;
    }
    .status-badge-ok {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-warn {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-alert {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_baseline_model():
    """Cache and load baseline model."""
    if BASELINE_CONFIG.model_path.exists():
        try:
            return BaselineToxicityClassifier.load(BASELINE_CONFIG.model_path), "TRAINED"
        except Exception as e:
            return None, f"Error: {e}"
    return None, "NOT_FOUND"


@st.cache_resource
def load_transformer_model():
    """Cache and load fine-tuned XLM-RoBERTa model."""
    if TRANSFORMER_CONFIG.model_dir.exists():
        try:
            from src.transformer_model import XLMRobertaToxicityClassifier
            model = XLMRobertaToxicityClassifier(model_name_or_path=TRANSFORMER_CONFIG.model_dir)
            model.load()
            return model, "TRAINED"
        except Exception as e:
            return None, f"Error: {e}"
    return None, "NOT_FOUND"


def heuristic_demo_scorer(text: str) -> float:
    """Heuristic fallback strictly for UI preview when no trained weights exist.
    
    Academic Integrity Notice: This is NOT a trained model. It is transparently
    flagged as a demonstration mock in the UI.
    """
    toxic_keywords = {
        "idiot", "stupid", "hate", "trash", "loser", "shut up", "imbécil",
        "asco", "dégage", "merde", "schwachkopf", "fresse", "aptal", "rezil",
        "disgusting", "kill", "die", "pathetic"
    }
    lowered = text.lower()
    matches = sum(1 for kw in toxic_keywords if kw in lowered)
    if matches == 0:
        return 0.08
    return min(0.96, 0.45 + (matches * 0.18))


# Multilingual test examples
EXAMPLE_COMMENTS = [
    ("English (Non-toxic)", "This is a very thoughtful, well-researched article. Thank you for clarifying this topic!"),
    ("English (Toxic)", "You are a complete idiot and a total loser, shut your mouth!"),
    ("Spanish (Non-toxic)", "No estoy totalmente de acuerdo con el autor, pero agradezco su perspectiva tan detallada."),
    ("Spanish (Toxic)", "Eres un imbécil total y das asco, cállate la boca."),
    ("French (Non-toxic)", "Merci beaucoup pour cette analyse rigoureuse et très instructive."),
    ("French (Toxic)", "Tu es vraiment un pauvre idiot bon à rien, dégage d'ici."),
    ("German (Non-toxic)", "Der Bericht fasst die wesentlichen Aspekte sachlich und objektiv zusammen."),
    ("German (Toxic)", "Du bist so ein verdammter Idiot, halt einfach deine dumme Fresse."),
    ("Italian (Non-toxic)", "Questo articolo spiega il contesto in maniera impeccabile e molto costruttiva."),
    ("Italian (Toxic)", "Sei un perfetto idiota senza cervello, vergognati e vattene via."),
    ("Turkish (Non-toxic)", "Bu makale çok bilgilendirici olmuş, emeğinize sağlık teşekkür ederim."),
    ("Turkish (Toxic)", "Sen tam bir aptalsın, hiçbir şeyden anlamıyorsun defol git buradan."),
]


# ==============================================================================
# SIDEBAR CONFIGURATION
# ==============================================================================
st.sidebar.title("🛡️ Classifier Settings")

# Model Selection
model_choice = st.sidebar.selectbox(
    "Select Model Architecture",
    ["Baseline (TF-IDF + Logistic Regression)", "XLM-RoBERTa (Multilingual Transformer)"],
)

threshold = st.sidebar.slider(
    "Toxicity Decision Threshold",
    min_value=0.05,
    max_value=0.95,
    value=APP_CONFIG.default_threshold,
    step=0.05,
    help="Scores at or above this threshold will be flagged as Toxic. Lower values increase recall (fewer missed toxic comments), higher values increase precision (fewer false alarms).",
)

# Hardware and Model Status Detection
device = get_default_device()
st.sidebar.markdown(f"**Compute Hardware:** `{device.type.upper()}`")

baseline_model, baseline_status = load_baseline_model()
transformer_model, transformer_status = load_transformer_model()

selected_model = None
is_demo_fallback = False

if "Baseline" in model_choice:
    if baseline_status == "TRAINED":
        selected_model = baseline_model
        st.sidebar.markdown('<span class="status-badge-ok">● Baseline: Ready</span>', unsafe_allow_html=True)
    else:
        is_demo_fallback = True
        st.sidebar.markdown('<span class="status-badge-warn">⚠️ Baseline: File Missing</span>', unsafe_allow_html=True)
        st.sidebar.caption("Run `python train.py --model baseline --use-synthetic` to generate model weights.")
else:
    if transformer_status == "TRAINED":
        selected_model = transformer_model
        st.sidebar.markdown('<span class="status-badge-ok">● Transformer: Ready</span>', unsafe_allow_html=True)
    else:
        is_demo_fallback = True
        st.sidebar.markdown('<span class="status-badge-warn">⚠️ Transformer: File Missing</span>', unsafe_allow_html=True)
        st.sidebar.caption("Weights not found in `models/xlm-roberta-toxic/`. See Colab instructions in `notebooks/`.")

st.sidebar.divider()
st.sidebar.markdown("""
**Academic Disclaimer**:
Zero-shot cross-lingual toxicity detection carries risks of cultural bias and dialect disparity. False positives may occur on sensitive keywords, while sarcasm may lead to false negatives.
""")


# ==============================================================================
# MAIN PAGE NAVIGATION
# ==============================================================================
st.title("Multilingual Toxic Comment Classification")
st.markdown("NLP System for cross-lingual toxicity detection using **TF-IDF Char/Word n-grams** and **XLM-RoBERTa**.")

if is_demo_fallback:
    st.warning(
        "⚠️ **DEMO PREVIEW MODE**: The requested trained model weights are not present in `models/`. "
        "The application is operating with a heuristic rule-based demonstration fallback. "
        "To evaluate real trained weights, run `python train.py --model baseline` (locally) or follow `notebooks/colab_training.ipynb` (GPU)."
    )

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Single Comment Analysis",
    "📁 Batch CSV Classification",
    "📊 Benchmarks & Evaluation",
    "📖 Model Architecture & Ethics",
])


# ==============================================================================
# TAB 1: SINGLE COMMENT ANALYSIS
# ==============================================================================
with tab1:
    st.subheader("Analyze Single Comment")

    # Example picker
    selected_example_label = st.selectbox(
        "Load a Multilingual Example Comment:",
        ["-- Select an example --"] + [f"{label}: {text[:50]}..." for label, text in EXAMPLE_COMMENTS],
    )
    
    initial_text = ""
    if selected_example_label != "-- Select an example --":
        index = [f"{l}: {t[:50]}..." for l, t in EXAMPLE_COMMENTS].index(selected_example_label)
        initial_text = EXAMPLE_COMMENTS[index][1]

    user_input = st.text_area(
        "Enter comment text (in any supported language):",
        value=initial_text,
        height=110,
        placeholder="Type or paste a comment here...",
    )

    col1, col2 = st.columns([1, 4])
    analyze_btn = col1.button("Classify Comment", type="primary")

    if analyze_btn or user_input:
        cleaned = clean_text(user_input)
        if not cleaned:
            st.error("Input text is empty or contains only whitespace/unprintable characters.")
        else:
            # Score computation
            if not is_demo_fallback and selected_model is not None:
                probs = selected_model.predict_proba([cleaned])[0]
                p_toxic = float(probs[1])
            else:
                p_toxic = heuristic_demo_scorer(cleaned)

            p_non_toxic = 1.0 - p_toxic
            is_toxic = p_toxic >= threshold

            st.divider()
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                if is_toxic:
                    st.error(f"### ⚠️ TOXIC COMMENT DETECTED\nDecision Threshold: `{threshold:.2f}`")
                else:
                    st.success(f"### ✅ NON-TOXIC COMMENT\nDecision Threshold: `{threshold:.2f}`")

                st.markdown(f"**Toxicity Probability:** `{p_toxic:.4f}` ({p_toxic * 100:.1f}%)")
                st.progress(float(p_toxic))

                st.markdown(f"**Non-Toxicity Probability:** `{p_non_toxic:.4f}` ({p_non_toxic * 100:.1f}%)")
                st.progress(float(p_non_toxic))

            with res_col2:
                st.markdown("#### Preprocessing Normalization")
                st.text(f"Original Length: {len(user_input)} chars")
                st.text(f"Cleaned Text:   {cleaned}")
                st.caption("Normalized URLs, user mentions, control chars, and HTML entities while preserving true casing, emojis, and punctuation.")

                if is_toxic and p_toxic < 0.65:
                    st.info("💡 **Borderline Toxic**: The score is close to the threshold. Consider human moderation review.")
                elif not is_toxic and p_toxic > 0.35:
                    st.info("💡 **Borderline Non-Toxic**: Moderate risk of subtle sarcasm or masked hostility.")


# ==============================================================================
# TAB 2: BATCH CSV CLASSIFICATION
# ==============================================================================
with tab2:
    st.subheader("Batch Dataset Classification")
    st.markdown("Upload a CSV file containing comments to classify multiple records simultaneously.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(batch_df)}** rows. Available columns: `{list(batch_df.columns)}`")

            # Column selector
            text_candidates = [c for c in batch_df.columns if "text" in c.lower() or "comment" in c.lower()]
            default_col = text_candidates[0] if text_candidates else batch_df.columns[0]
            chosen_col = st.selectbox("Select column containing comment text:", batch_df.columns, index=list(batch_df.columns).index(default_col))

            if st.button("Run Batch Inference", type="primary"):
                with st.spinner("Processing comments..."):
                    cleaned_texts = [clean_text(t) for t in batch_df[chosen_col]]
                    
                    if not is_demo_fallback and selected_model is not None:
                        probs = selected_model.predict_proba(cleaned_texts)[:, 1]
                    else:
                        probs = np.array([heuristic_demo_scorer(t) for t in cleaned_texts])

                    batch_df["cleaned_text"] = cleaned_texts
                    batch_df["toxicity_prob"] = probs
                    batch_df["prediction"] = (probs >= threshold).astype(int)
                    batch_df["verdict"] = np.where(batch_df["prediction"] == 1, "Toxic", "Non-toxic")

                toxic_count = int(np.sum(batch_df["prediction"] == 1))
                non_toxic_count = int(np.sum(batch_df["prediction"] == 0))

                bcol1, bcol2, bcol3 = st.columns(3)
                bcol1.metric("Total Comments", len(batch_df))
                bcol2.metric("Flagged Toxic", toxic_count, f"{(toxic_count/len(batch_df))*100:.1f}%")
                bcol3.metric("Non-Toxic", non_toxic_count, f"{(non_toxic_count/len(batch_df))*100:.1f}%")

                st.dataframe(batch_df[[chosen_col, "verdict", "toxicity_prob"]].head(100), use_container_width=True)

                csv_data = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Classified CSV",
                    data=csv_data,
                    file_name="toxicity_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error reading or processing CSV: {e}")


# ==============================================================================
# TAB 3: BENCHMARKS & EVALUATION
# ==============================================================================
with tab3:
    st.subheader("Empirical Model Evaluation & Comparison")
    st.markdown("Visualizations and quantitative metrics on held-out evaluation data.")

    baseline_metrics_path = RESULTS_DIR / "baseline_tf-idf_metrics.json"
    transformer_metrics_path = RESULTS_DIR / "xlm-roberta_metrics.json"

    metrics_loaded = {}
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path, "r", encoding="utf-8") as f:
            metrics_loaded["Baseline TF-IDF"] = json.load(f)

    if transformer_metrics_path.exists():
        with open(transformer_metrics_path, "r", encoding="utf-8") as f:
            metrics_loaded["XLM-RoBERTa"] = json.load(f)

    if metrics_loaded:
        st.markdown("### Model Performance Comparison")
        rows = []
        for name, m in metrics_loaded.items():
            rows.append({
                "Model": name,
                "Accuracy": f"{m.get('accuracy', 0):.4f}",
                "Precision": f"{m.get('precision', 0):.4f}",
                "Recall": f"{m.get('recall', 0):.4f}",
                "Toxic F1": f"{m.get('f1', 0):.4f}",
                "Macro F1": f"{m.get('macro_f1', 0):.4f}",
                "ROC-AUC": f"{m['roc_auc']:.4f}" if m.get("roc_auc") is not None else "N/A",
                "PR-AUC": f"{m['pr_auc']:.4f}" if m.get("pr_auc") is not None else "N/A",
            })
        st.table(pd.DataFrame(rows))

        # Plots
        st.markdown("### Evaluation Visualizations")
        pcol1, pcol2 = st.columns(2)

        for name, m in metrics_loaded.items():
            plots = m.get("plots", {})
            cm_img = plots.get("confusion_matrix")
            roc_img = plots.get("roc_curve")
            pr_img = plots.get("pr_curve")

            if cm_img and Path(cm_img).exists():
                pcol1.image(cm_img, caption=f"Confusion Matrix: {name}", use_container_width=True)
            if roc_img and Path(roc_img).exists():
                pcol2.image(roc_img, caption=f"ROC Curve: {name}", use_container_width=True)

        # Per-language breakdown if present
        for name, m in metrics_loaded.items():
            per_lang = m.get("per_language")
            if per_lang:
                st.markdown(f"#### Per-Language Breakdown ({name})")
                st.dataframe(pd.DataFrame(per_lang), use_container_width=True)

    else:
        st.info(
            "ℹ️ No saved evaluation metrics found in `results/`.\n\n"
            "To generate evaluation results on test data, run:\n"
            "```bash\n"
            "python evaluate.py --model baseline --use-synthetic\n"
            "```\n"
            "This will produce `results/baseline_tf-idf_metrics.json` and generate confusion matrix and ROC curve plots."
        )


# ==============================================================================
# TAB 4: MODEL ARCHITECTURE & ETHICS
# ==============================================================================
with tab4:
    st.subheader("Model Architecture, Zero-Shot Transfer & Ethical Considerations")

    st.markdown("""
### 1. Methodology Comparison

| Feature | Baseline Model | Deep Transformer |
|---|---|---|
| **Architecture** | Word + Char n-gram TF-IDF + Logistic Regression | FacebookAI/xlm-roberta-base (12-layer Transformer) |
| **Vocabulary** | Lexical tokens + Subword char n-grams (2-5 chars) | 250,000 SentencePiece multilingual subwords |
| **Cross-Lingual Ability** | Shared Latin character sequences only | Shared multilingual vector space across 100 languages |
| **Obfuscation Robustness** | Moderate (char n-grams catch typos/leetspeak) | High (Byte-level SentencePiece subwords) |
| **Inference Latency** | **< 1 millisecond** (CPU lightweight) | ~20-50 milliseconds (GPU/MPS required for high throughput) |

---

### 2. Zero-Shot Cross-Lingual Transfer
In academic multilingual toxicity classification (such as the Jigsaw dataset), high-volume human annotations exist primarily in English, while testing occurs on Italian, Spanish, French, Russian, Turkish, and Portuguese.
- **Hypothesis**: Masked language models pre-trained on diverse multilingual corpora align conceptual representations of toxicity across languages.
- **Reality & Pitfalls**: Idiomatic toxicity, localized cultural slurs, and grammatical negation patterns may not transfer uniformly, resulting in lower recall on non-English evaluation splits.

---

### 3. Ethical Considerations & Bias Risks
- **False Positives on Identity Terms**: Toxicity detectors historically over-flag comments containing identity terms (e.g. *gay*, *lesbian*, *muslim*, *black*, *transgender*) even in purely benign, affirmative contexts.
- **Reclaimed Slurs**: In-group vernacular and reclaimed terminology present severe moderation ambiguities.
- **Dialectal Inequity**: Non-standard dialects and African American Vernacular English (AAVE) suffer higher false-positive toxicity rates.
- **Recommended Remediation**: Automated classifiers must serve as **decision support aids** to human moderators rather than fully autonomous censors.
""")
