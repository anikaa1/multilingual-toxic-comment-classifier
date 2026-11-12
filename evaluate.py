"""Model evaluation script.

Evaluates trained Baseline and/or Transformer models on held-out test data.
Generates metrics, comparison tables, error analysis, and visualization figures.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BASELINE_CONFIG,
    DATASET_CONFIG,
    TRANSFORMER_CONFIG,
    RESULTS_DIR,
)
from src.dataset import load_and_prepare_datasets
from src.baseline import BaselineToxicityClassifier
from src.metrics import evaluate_model_performance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate")


def evaluate_baseline(test_df: pd.DataFrame, threshold: float = 0.5) -> dict:
    """Evaluate Baseline TF-IDF + Logistic Regression."""
    logger.info("Evaluating Baseline (TF-IDF + Logistic Regression)...")
    if not BASELINE_CONFIG.model_path.exists():
        raise FileNotFoundError(
            f"Baseline model file not found at: {BASELINE_CONFIG.model_path}\n"
            "Run 'python train.py --model baseline' first."
        )

    classifier = BaselineToxicityClassifier.load(BASELINE_CONFIG.model_path)
    probs = classifier.predict_proba(test_df["clean_text"].tolist())
    preds = (probs[:, 1] >= threshold).astype(int)

    metrics = evaluate_model_performance(
        model_name="Baseline TF-IDF",
        test_df=test_df,
        predictions=preds,
        probabilities=probs,
        save_dir=RESULTS_DIR,
        lang_col="lang",
    )
    return metrics


def evaluate_transformer(test_df: pd.DataFrame, threshold: float = 0.5) -> dict:
    """Evaluate Fine-Tuned XLM-RoBERTa."""
    logger.info("Evaluating Transformer (XLM-RoBERTa)...")
    if not TRANSFORMER_CONFIG.model_dir.exists():
        raise FileNotFoundError(
            f"Transformer model directory not found at: {TRANSFORMER_CONFIG.model_dir}\n"
            "Run 'python train.py --model transformer' first, or follow Colab training instructions."
        )

    try:
        from src.transformer_model import XLMRobertaToxicityClassifier
    except ImportError as e:
        logger.error(f"Cannot import transformer model: {e}")
        sys.exit(1)

    classifier = XLMRobertaToxicityClassifier(
        model_name_or_path=TRANSFORMER_CONFIG.model_dir,
        default_threshold=threshold,
    )
    classifier.load()

    probs = classifier.predict_proba(test_df["clean_text"].tolist())
    preds = (probs[:, 1] >= threshold).astype(int)

    metrics = evaluate_model_performance(
        model_name="XLM-RoBERTa",
        test_df=test_df,
        predictions=preds,
        probabilities=probs,
        save_dir=RESULTS_DIR,
        lang_col="lang",
    )
    return metrics


def generate_comparison_summary(all_metrics: dict, save_dir: Path):
    """Generate and display side-by-side comparison table."""
    rows = []
    for model_name, metrics in all_metrics.items():
        cm = metrics.get("confusion_matrix", {})
        roc = metrics.get("roc_auc")
        roc_str = f"{roc:.4f}" if roc is not None else "N/A"
        rows.append({
            "Model": model_name,
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Precision": f"{metrics['precision']:.4f}",
            "Recall": f"{metrics['recall']:.4f}",
            "Toxic F1": f"{metrics['f1']:.4f}",
            "Macro F1": f"{metrics['macro_f1']:.4f}",
            "ROC-AUC": roc_str,
            "TN": cm.get("tn", 0),
            "FP": cm.get("fp", 0),
            "FN": cm.get("fn", 0),
            "TP": cm.get("tp", 0),
        })

    summary_df = pd.DataFrame(rows)
    csv_path = save_dir / "models_comparison_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80 + "\n")
    logger.info(f"Comparison summary saved to {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Toxic Comment Classifiers")
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "transformer", "all"],
        default="all",
        help="Which model(s) to evaluate",
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Use synthetic dataset for verification testing",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="Custom path to test dataset CSV",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification decision threshold (default: 0.5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        _, _, test_df = load_and_prepare_datasets(
            test_path=args.test_path,
            use_synthetic=args.use_synthetic,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Loaded evaluation test set: {len(test_df)} samples.")
    if "label" not in test_df.columns:
        logger.error("Test dataset must contain ground truth labels for evaluation.")
        sys.exit(1)

    all_metrics = {}

    if args.model in ("baseline", "all"):
        try:
            baseline_metrics = evaluate_baseline(test_df, threshold=args.threshold)
            all_metrics["Baseline TF-IDF"] = baseline_metrics
        except Exception as e:
            logger.warning(f"Could not evaluate Baseline: {e}")

    if args.model in ("transformer", "all"):
        try:
            transformer_metrics = evaluate_transformer(test_df, threshold=args.threshold)
            all_metrics["XLM-RoBERTa"] = transformer_metrics
        except Exception as e:
            logger.warning(f"Could not evaluate Transformer: {e}")

    if all_metrics:
        generate_comparison_summary(all_metrics, RESULTS_DIR)
    else:
        logger.error("No models were successfully evaluated. Please train a model first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
