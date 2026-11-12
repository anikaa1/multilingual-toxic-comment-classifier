"""TF-IDF and Logistic Regression baseline model for toxic comment classification."""

import logging
from pathlib import Path
from typing import List, Optional, Union
import joblib
import numpy as np
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import BASELINE_CONFIG

logger = logging.getLogger(__name__)


class BaselineToxicityClassifier:
    """Baseline toxicity classifier combining Word & Character n-gram TF-IDF with Logistic Regression.
    
    Academic Rationale:
    1. Word n-grams capture common toxic vocabulary and short phrases (e.g., 'shut up', 'hate you').
    2. Character n-grams (2-5 chars) capture morphological roots, subword affixes, and intentional
       obfuscations / typos (e.g., 'a$$hole', 'f**k', 'd-i-e') that defeat standard tokenizers.
    3. Character n-grams also provide cross-lingual signal across related Latin-script languages.
    4. Class weighting ('balanced') addresses severe class imbalance typical of toxicity datasets.
    """

    def __init__(
        self,
        word_ngram_range=(1, 2),
        word_max_features=15000,
        char_ngram_range=(2, 5),
        char_max_features=25000,
        C: float = 2.0,
        max_iter: int = 1000,
        class_weight: Optional[str] = "balanced",
        random_seed: int = 42,
    ):
        self.word_ngram_range = word_ngram_range
        self.word_max_features = word_max_features
        self.char_ngram_range = char_ngram_range
        self.char_max_features = char_max_features
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_seed = random_seed

        # Build pipeline
        self.pipeline: Optional[Pipeline] = None
        self._is_fitted = False

    def _build_pipeline(self) -> Pipeline:
        """Construct the scikit-learn FeatureUnion and LogisticRegression pipeline."""
        features = FeatureUnion([
            (
                "word_tfidf",
                TfidfVectorizer(
                    ngram_range=self.word_ngram_range,
                    max_features=self.word_max_features,
                    sublinear_tf=True,
                    strip_accents=None,
                    lowercase=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    ngram_range=self.char_ngram_range,
                    analyzer="char",
                    max_features=self.char_max_features,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
        ])

        clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_seed,
            solver="lbfgs",
            n_jobs=-1,
        )

        return Pipeline([
            ("features", features),
            ("classifier", clf),
        ])

    def fit(self, texts: List[str], labels: List[int]) -> "BaselineToxicityClassifier":
        """Train the TF-IDF feature extractor and logistic regression classifier."""
        logger.info(f"Training Baseline TF-IDF + Logistic Regression on {len(texts)} samples...")
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(texts, labels)
        self._is_fitted = True
        logger.info("Baseline training complete.")
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Return probability estimates for [Non-toxic (0), Toxic (1)]."""
        if not self._is_fitted or self.pipeline is None:
            raise RuntimeError("Baseline model has not been trained or loaded.")
        return self.pipeline.predict_proba(texts)

    def predict(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions using the specified decision threshold."""
        probabilities = self.predict_proba(texts)
        toxic_probs = probabilities[:, 1]
        return (toxic_probs >= threshold).astype(int)

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Persist fitted pipeline to disk."""
        if not self._is_fitted or self.pipeline is None:
            raise RuntimeError("Cannot save an unfitted baseline model.")
        save_path = Path(path) if path else BASELINE_CONFIG.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, save_path)
        logger.info(f"Saved baseline model to {save_path}")
        return save_path

    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> "BaselineToxicityClassifier":
        """Load trained baseline model from disk."""
        load_path = Path(path) if path else BASELINE_CONFIG.model_path
        if not load_path.exists():
            raise FileNotFoundError(f"Baseline model file not found at: {load_path}")
        instance = cls()
        instance.pipeline = joblib.load(load_path)
        instance._is_fitted = True
        logger.info(f"Loaded baseline model from {load_path}")
        return instance
