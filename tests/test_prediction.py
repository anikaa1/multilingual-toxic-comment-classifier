"""Unit tests for model training, prediction, and threshold application."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.baseline import BaselineToxicityClassifier
from src.dataset import load_and_prepare_datasets


class TestPrediction(unittest.TestCase):
    """Test suite for baseline model fitting, probability calibration, and predictions."""

    @classmethod
    def setUpClass(cls):
        """Load synthetic smoke dataset for quick model fitting test."""
        cls.train_df, cls.val_df, cls.test_df = load_and_prepare_datasets(use_synthetic=True, random_seed=42)
        cls.model = BaselineToxicityClassifier(
            word_ngram_range=(1, 2),
            word_max_features=500,
            char_ngram_range=(2, 4),
            char_max_features=500,
            C=1.0,
            max_iter=200,
        )
        cls.model.fit(cls.train_df["clean_text"].tolist(), cls.train_df["label"].tolist())

    def test_predict_proba_shape_and_bounds(self):
        """Probability outputs must have shape (N, 2) and sum to 1.0."""
        texts = ["This is great!", "You are a total loser and an idiot."]
        probs = self.model.predict_proba(texts)

        self.assertEqual(probs.shape, (2, 2))
        for row in probs:
            self.assertTrue(np.all(row >= 0.0) and np.all(row <= 1.0))
            self.assertAlmostEqual(np.sum(row), 1.0, places=5)

    def test_predict_threshold_effect(self):
        """Higher threshold should produce fewer or equal positive predictions."""
        texts = self.test_df["clean_text"].tolist()
        preds_low = self.model.predict(texts, threshold=0.20)
        preds_high = self.model.predict(texts, threshold=0.80)

        self.assertGreaterEqual(np.sum(preds_low == 1), np.sum(preds_high == 1))

    def test_model_persistence(self):
        """Trained baseline model must save to and load from disk cleanly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_baseline.joblib"
            self.model.save(tmp_path)
            self.assertTrue(tmp_path.exists())

            loaded_model = BaselineToxicityClassifier.load(tmp_path)
            texts = ["Testing save and reload."]
            orig_probs = self.model.predict_proba(texts)
            loaded_probs = loaded_model.predict_proba(texts)

            np.testing.assert_allclose(orig_probs, loaded_probs, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
