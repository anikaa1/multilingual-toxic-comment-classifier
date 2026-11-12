"""Comprehensive academic evaluation metrics, visualization generators, and error analysis."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Set writable cache dir for matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_prob: Optional[Union[List[float], np.ndarray]] = None,
) -> Dict[str, Any]:
    """Compute rigorous classification metrics for toxicity evaluation.
    
    Returns standard metrics:
    - Precision (toxic class 1)
    - Recall (toxic class 1)
    - F1-score (toxic class 1)
    - Macro-F1 & Weighted-F1
    - Accuracy
    - ROC-AUC and PR-AUC (Average Precision) if probabilities are provided
    - Confusion Matrix counts (TN, FP, FN, TP)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "support": {
            "non_toxic": int(np.sum(y_true == 0)),
            "toxic": int(np.sum(y_true == 1)),
            "total": len(y_true),
        },
    }

    # Compute ROC-AUC and PR-AUC if probabilities are available and both classes exist in y_true
    if y_prob is not None:
        y_prob = np.asarray(y_prob).astype(float)
        unique_classes = np.unique(y_true)
        if len(unique_classes) > 1:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
                metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC or PR-AUC: {e}")
                metrics["roc_auc"] = None
                metrics["pr_auc"] = None
        else:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
            logger.info("ROC-AUC not defined because only one class is present in target.")
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def find_optimal_threshold(
    y_true: Union[List[int], np.ndarray],
    y_prob: Union[List[float], np.ndarray],
    metric: str = "f1",
) -> Tuple[float, float]:
    """Find the optimal classification decision threshold on validation data.
    
    Academic rationale:
    The default threshold of 0.5 assumes equal cost between false positives and
    false negatives and balanced priors. In severe class imbalance or sensitive moderation,
    threshold tuning on the validation partition maximizes target utility (e.g., F1)
    without causing test-set data leakage.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh = 0.50
    best_score = -1.0

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, preds, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, preds, zero_division=0)
        else:
            score = accuracy_score(y_true, preds)

        if score > best_score:
            best_score = score
            best_thresh = float(t)

    logger.info(f"Threshold calibration: Optimal threshold for {metric} is {best_thresh:.3f} (Score={best_score:.4f})")
    return best_thresh, float(best_score)


def compute_per_language_metrics(
    df: pd.DataFrame,
    target_col: str = "label",
    pred_col: str = "pred",
    prob_col: Optional[str] = "prob",
    lang_col: str = "lang",
) -> pd.DataFrame:
    """Breakdown classification metrics by language to evaluate cross-lingual transfer."""
    if lang_col not in df.columns or df[lang_col].nunique() <= 1:
        logger.info("Language column missing or single-language only; skipping per-language breakdown.")
        return pd.DataFrame()

    records = []
    for lang, group in df.groupby(lang_col):
        y_true = group[target_col].values
        y_pred = group[pred_col].values
        y_prob = group[prob_col].values if (prob_col and prob_col in group.columns) else None

        lang_metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        records.append({
            "lang": lang,
            "sample_count": len(group),
            "toxic_count": int(np.sum(y_true == 1)),
            "accuracy": lang_metrics["accuracy"],
            "precision": lang_metrics["precision"],
            "recall": lang_metrics["recall"],
            "f1": lang_metrics["f1"],
            "roc_auc": lang_metrics["roc_auc"],
        })

    result_df = pd.DataFrame(records).sort_values(by="sample_count", ascending=False).reset_index(drop=True)
    return result_df


def extract_error_cases(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    target_col: str = "label",
    pred_col: str = "pred",
    prob_col: str = "prob",
    lang_col: str = "lang",
    n: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """Identify top False Positives and False Negatives for qualitative error analysis.
    
    - False Positives: Actual Non-toxic (0) but predicted Toxic (1) with highest probability.
    - False Negatives: Actual Toxic (1) but predicted Non-toxic (0) with lowest probability.
    """
    df = df.copy()
    has_prob = prob_col in df.columns
    has_lang = lang_col in df.columns

    # False Positives
    fps = df[(df[target_col] == 0) & (df[pred_col] == 1)]
    if has_prob:
        fps = fps.sort_values(by=prob_col, ascending=False)
    top_fps = []
    for _, row in fps.head(n).iterrows():
        top_fps.append({
            "text": str(row[text_col]),
            "true_label": 0,
            "pred_label": 1,
            "prob_toxic": float(row[prob_col]) if has_prob else None,
            "lang": str(row[lang_col]) if has_lang else "unknown",
        })

    # False Negatives
    fns = df[(df[target_col] == 1) & (df[pred_col] == 0)]
    if has_prob:
        fns = fns.sort_values(by=prob_col, ascending=True)
    top_fns = []
    for _, row in fns.head(n).iterrows():
        top_fns.append({
            "text": str(row[text_col]),
            "true_label": 1,
            "pred_label": 0,
            "prob_toxic": float(row[prob_col]) if has_prob else None,
            "lang": str(row[lang_col]) if has_lang else "unknown",
        })

    return {
        "false_positives": top_fps,
        "false_negatives": top_fns,
    }


# Visualization Functions

def plot_confusion_matrix(cm_dict: Dict[str, int], model_name: str, save_path: Path) -> Path:
    """Plot and save confusion matrix heatmap."""
    matrix = np.array([
        [cm_dict["tn"], cm_dict["fp"]],
        [cm_dict["fn"], cm_dict["tp"]],
    ])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-toxic (0)", "Toxic (1)"],
        yticklabels=["Non-toxic (0)", "Toxic (1)"],
    )
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted Class")
    plt.ylabel("Ground Truth Class")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def plot_roc_curve_figure(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    save_path: Path,
) -> Optional[Path]:
    """Plot and save ROC curve."""
    if len(np.unique(y_true)) < 2:
        return None

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_score = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"{model_name} (AUC = {roc_score:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic (ROC): {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def plot_precision_recall_curve_figure(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    save_path: Path,
) -> Optional[Path]:
    """Plot and save Precision-Recall curve."""
    if len(np.unique(y_true)) < 2:
        return None

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_prec = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="#ff7f0e", lw=2, label=f"{model_name} (AP = {avg_prec:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve: {model_name}")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def evaluate_model_performance(
    model_name: str,
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray],
    save_dir: Path,
    lang_col: str = "lang",
) -> Dict[str, Any]:
    """Run complete evaluation suite, save metrics JSON and plots."""
    save_dir.mkdir(parents=True, exist_ok=True)
    y_true = test_df["label"].values
    toxic_probs = probabilities[:, 1] if probabilities is not None else None

    # Base classification metrics
    metrics = compute_classification_metrics(y_true, predictions, toxic_probs)
    metrics["model_name"] = model_name

    # Per-language breakdown
    eval_df = test_df.copy()
    eval_df["pred"] = predictions
    if toxic_probs is not None:
        eval_df["prob"] = toxic_probs

    lang_metrics_df = compute_per_language_metrics(eval_df, target_col="label", pred_col="pred", prob_col="prob", lang_col=lang_col)
    if not lang_metrics_df.empty:
        metrics["per_language"] = lang_metrics_df.to_dict(orient="records")
        lang_csv_path = save_dir / f"{model_name.lower().replace(' ', '_')}_per_language.csv"
        lang_metrics_df.to_csv(lang_csv_path, index=False)

    # Error analysis
    error_cases = extract_error_cases(eval_df, target_col="label", pred_col="pred", prob_col="prob", lang_col=lang_col)
    metrics["error_cases"] = error_cases

    # Visualizations
    cm_path = save_dir / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    plot_confusion_matrix(metrics["confusion_matrix"], model_name, cm_path)
    metrics["plots"] = {"confusion_matrix": str(cm_path)}

    if toxic_probs is not None and len(np.unique(y_true)) > 1:
        roc_path = save_dir / f"{model_name.lower().replace(' ', '_')}_roc_curve.png"
        pr_path = save_dir / f"{model_name.lower().replace(' ', '_')}_pr_curve.png"
        plot_roc_curve_figure(y_true, toxic_probs, model_name, roc_path)
        plot_precision_recall_curve_figure(y_true, toxic_probs, model_name, pr_path)
        metrics["plots"]["roc_curve"] = str(roc_path)
        metrics["plots"]["pr_curve"] = str(pr_path)

    # Save metrics JSON
    metrics_json_path = save_dir / f"{model_name.lower().replace(' ', '_')}_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Evaluation complete for {model_name}. Metrics saved to {metrics_json_path}")
    return metrics
