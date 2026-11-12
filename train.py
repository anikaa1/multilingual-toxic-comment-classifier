"""Command-line training script for Multilingual Toxic Comment Classifier.

Supports:
- Baseline (TF-IDF + Logistic Regression)
- XLM-RoBERTa Transformer fine-tuning
- Synthetic smoke test mode for fast environment verification
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BASELINE_CONFIG,
    DATASET_CONFIG,
    TRANSFORMER_CONFIG,
    RESULTS_DIR,
    get_default_device,
)
from src.dataset import load_and_prepare_datasets
from src.baseline import BaselineToxicityClassifier
from src.metrics import find_optimal_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train")


def train_baseline_pipeline(train_df, val_df, tune_threshold: bool = True):
    """Train and save TF-IDF + Logistic Regression baseline."""
    logger.info("=== Starting Baseline Training (TF-IDF + Logistic Regression) ===")
    classifier = BaselineToxicityClassifier(
        word_ngram_range=BASELINE_CONFIG.word_ngram_range,
        word_max_features=BASELINE_CONFIG.word_max_features,
        char_ngram_range=BASELINE_CONFIG.char_ngram_range,
        char_max_features=BASELINE_CONFIG.char_max_features,
        C=BASELINE_CONFIG.C,
        max_iter=BASELINE_CONFIG.max_iter,
        class_weight=BASELINE_CONFIG.class_weight,
        random_seed=BASELINE_CONFIG.random_seed,
    )

    classifier.fit(train_df["clean_text"].tolist(), train_df["label"].tolist())
    model_path = classifier.save(BASELINE_CONFIG.model_path)
    logger.info(f"Baseline model persisted to: {model_path}")

    # Threshold calibration on validation set
    if tune_threshold and len(val_df) > 0 and len(val_df["label"].unique()) > 1:
        val_probs = classifier.predict_proba(val_df["clean_text"].tolist())[:, 1]
        best_thresh, best_f1 = find_optimal_threshold(val_df["label"].tolist(), val_probs, metric="f1")
        logger.info(f"Validation threshold calibration: optimal threshold = {best_thresh:.3f} (Val F1 = {best_f1:.4f})")

    return classifier


def train_transformer_pipeline(train_df, val_df, args):
    """Train and save XLM-RoBERTa fine-tuned transformer."""
    logger.info("=== Starting Transformer Fine-Tuning (XLM-RoBERTa) ===")
    try:
        from src.transformer_model import train_transformer_model
    except ImportError as e:
        logger.error(
            f"PyTorch or Transformers import failed: {e}\n"
            "Please verify dependencies in requirements.txt are installed."
        )
        sys.exit(1)

    device = get_default_device()
    logger.info(f"Detected training hardware: {device}")

    metadata = train_transformer_model(
        train_texts=train_df["clean_text"].tolist(),
        train_labels=train_df["label"].tolist(),
        val_texts=val_df["clean_text"].tolist(),
        val_labels=val_df["label"].tolist(),
        output_dir=TRANSFORMER_CONFIG.model_dir,
        pretrained_model_name=args.model_name or TRANSFORMER_CONFIG.pretrained_model_name,
        learning_rate=args.lr or TRANSFORMER_CONFIG.learning_rate,
        epochs=args.epochs or TRANSFORMER_CONFIG.epochs,
        batch_size=args.batch_size or TRANSFORMER_CONFIG.batch_size,
        max_length=args.max_length or TRANSFORMER_CONFIG.max_length,
        seed=args.seed or TRANSFORMER_CONFIG.random_seed,
        handle_class_imbalance=not args.no_class_weights,
    )

    logger.info("Transformer training successfully completed.")
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multilingual Toxic Comment Classifiers")
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "transformer", "all"],
        default="baseline",
        help="Which model architecture to train (default: baseline)",
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Use small synthetic multi-lingual dataset for quick pipeline verification and testing",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Custom path to training CSV",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default=None,
        help="Custom path to validation CSV",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit number of rows for fast iteration",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Hugging Face model identifier (default: FacebookAI/xlm-roberta-base)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Epochs for transformer training")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for transformer")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate for transformer")
    parser.add_argument("--max-length", type=int, default=None, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random reproducibility seed")
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse class weighting in loss function",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load datasets
    try:
        train_df, val_df, test_df = load_and_prepare_datasets(
            train_path=args.train_path,
            val_path=args.val_path,
            use_synthetic=args.use_synthetic,
            sample_size=args.sample_size,
            random_seed=args.seed,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Loaded splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    if args.model in ("baseline", "all"):
        train_baseline_pipeline(train_df, val_df)

    if args.model in ("transformer", "all"):
        train_transformer_pipeline(train_df, val_df, args)

    logger.info("Training script completed successfully.")


if __name__ == "__main__":
    main()
