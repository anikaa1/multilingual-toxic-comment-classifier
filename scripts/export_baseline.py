#!/usr/bin/env python3
"""Export baseline model and preprocessing artifacts to JSON for in-browser JavaScript inference.

Exports:
1. Preprocessing normalization rules from preprocess.py
2. Word TF-IDF vectorizer vocabulary, IDF values, and settings
3. Char TF-IDF vectorizer vocabulary, IDF values, and settings
4. Logistic regression coefficients and intercept
"""

import json
import logging
from pathlib import Path
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def export_baseline(
    model_path: Path = Path("models/baseline_model.joblib"),
    output_path: Path = Path("docs/model/baseline.json"),
) -> None:
    """Load baseline pipeline and write complete inference payload to JSON."""
    if not model_path.exists():
        raise FileNotFoundError(f"Baseline model not found at {model_path}")

    logger.info(f"Loading baseline pipeline from {model_path}...")
    pipeline = joblib.load(model_path)

    features = pipeline.named_steps["features"]
    transformers = dict(features.transformer_list)
    word_vec = transformers["word_tfidf"]
    char_vec = transformers["char_tfidf"]
    clf = pipeline.named_steps["classifier"]

    w_size = len(word_vec.vocabulary_)
    c_size = len(char_vec.vocabulary_)
    logger.info(f"Extracting word features ({w_size}) and char features ({c_size})...")

    coef = clf.coef_[0]
    word_coef = [float(x) for x in coef[:w_size]]
    char_coef = [float(x) for x in coef[w_size:]]
    intercept = float(clf.intercept_[0])

    # Convert vocab keys to python int
    word_vocab = {k: int(v) for k, v in word_vec.vocabulary_.items()}
    char_vocab = {k: int(v) for k, v in char_vec.vocabulary_.items()}

    payload = {
        "metadata": {
            "name": "TF-IDF + Logistic Regression Baseline",
            "classes": ["Clean", "Toxic"],
            "threshold": 0.5,
        },
        "preprocessing": {
            "unescape_html": True,
            "strip_html_tags": True,
            "html_tag_regex": "<[^>]+>",
            "normalize_urls": True,
            "url_placeholder": "[URL]",
            "url_regex": r"https?://\S+|www\.\S+",
            "normalize_mentions": True,
            "mention_placeholder": "[USER]",
            "mention_regex": r"@[\w\-]+",
            "remove_control_chars": True,
            "control_chars_regex": r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
            "normalize_whitespace": True,
            "whitespace_regex": r"[ \t]+",
            "excessive_newlines_regex": r"\n\s*\n+",
        },
        "word_vectorizer": {
            "settings": {
                "ngram_range": list(word_vec.ngram_range),
                "sublinear_tf": bool(word_vec.sublinear_tf),
                "norm": word_vec.norm,
                "lowercase": bool(word_vec.lowercase),
                "analyzer": word_vec.analyzer,
                "token_pattern": word_vec.token_pattern,
            },
            "vocabulary": word_vocab,
            "idf": [float(x) for x in word_vec.idf_],
            "coef": word_coef,
        },
        "char_vectorizer": {
            "settings": {
                "ngram_range": list(char_vec.ngram_range),
                "sublinear_tf": bool(char_vec.sublinear_tf),
                "norm": char_vec.norm,
                "lowercase": bool(char_vec.lowercase),
                "analyzer": char_vec.analyzer,
            },
            "vocabulary": char_vocab,
            "idf": [float(x) for x in char_vec.idf_],
            "coef": char_coef,
        },
        "classifier": {
            "type": "LogisticRegression",
            "intercept": intercept,
            "threshold": 0.5,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing payload to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Baseline export complete: {output_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    export_baseline()
