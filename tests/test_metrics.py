"""Unit tests for classification metrics, threshold search, and evaluation functions."""

import unittest
import numpy as np
import pandas as pd
from src.metrics import (
    compute_classification_metrics,
    find_optimal_threshold,
    compute_per_language_metrics,
    extract_error_cases,
)


class TestMetrics(unittest.TestCase):
    """Test suite for metrics calculation."""

    def test_compute_classification_metrics_perfect(self):
        """Test metrics under perfect classification."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        y_prob = [0.1, 0.2, 0.8, 0.9]

        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["confusion_matrix"]["tp"], 2)
        self.assertEqual(metrics["confusion_matrix"]["tn"], 2)
        self.assertEqual(metrics["confusion_matrix"]["fp"], 0)
        self.assertEqual(metrics["confusion_matrix"]["fn"], 0)
        self.assertEqual(metrics["roc_auc"], 1.0)

    def test_compute_classification_metrics_imperfect(self):
        """Test metrics with mixed errors."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]  # 1 FP, 1 FN

        metrics = compute_classification_metrics(y_true, y_pred)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["confusion_matrix"]["fp"], 1)
        self.assertEqual(metrics["confusion_matrix"]["fn"], 1)
        self.assertEqual(metrics["confusion_matrix"]["tp"], 1)
        self.assertEqual(metrics["confusion_matrix"]["tn"], 1)

    def test_find_optimal_threshold(self):
        """Test calibration of decision threshold."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.2, 0.4, 0.45, 0.8])
        # If threshold is 0.5: preds are [0, 0, 0, 1] -> recall 0.5, f1 = 2*1*0.5/(1+0.5)=0.667
        # If threshold is 0.42: preds are [0, 0, 1, 1] -> perfect f1 = 1.0
        best_t, best_f1 = find_optimal_threshold(y_true, y_prob, metric="f1")
        self.assertTrue(0.40 <= best_t <= 0.45)
        self.assertAlmostEqual(best_f1, 1.0)

    def test_compute_per_language_metrics(self):
        """Test grouping and metrics breakdown by language."""
        df = pd.DataFrame({
            "label": [0, 1, 0, 1],
            "pred": [0, 1, 1, 1],
            "prob": [0.1, 0.9, 0.6, 0.8],
            "lang": ["en", "en", "es", "es"],
        })
        lang_df = compute_per_language_metrics(df, target_col="label", pred_col="pred", prob_col="prob", lang_col="lang")
        self.assertEqual(len(lang_df), 2)
        self.assertIn("lang", lang_df.columns)
        self.assertIn("f1", lang_df.columns)

    def test_extract_error_cases(self):
        """Test identification of false positives and false negatives."""
        df = pd.DataFrame({
            "clean_text": [
                "Kind compliment",
                "Harmless mention of word idiot in quote",
                "Subtle disguised threat",
                "Explicit violent insult",
            ],
            "label": [0, 0, 1, 1],
            "pred":  [0, 1, 0, 1],
            "prob":  [0.05, 0.85, 0.20, 0.95],
            "lang":  ["en", "en", "en", "en"],
        })
        cases = extract_error_cases(df, text_col="clean_text", target_col="label", pred_col="pred", prob_col="prob")
        self.assertEqual(len(cases["false_positives"]), 1)
        self.assertEqual(len(cases["false_negatives"]), 1)
        self.assertIn("idiot", cases["false_positives"][0]["text"])
        self.assertIn("threat", cases["false_negatives"][0]["text"])


if __name__ == "__main__":
    unittest.main()
