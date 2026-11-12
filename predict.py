"""Inference script for single-text and batch-file toxicity prediction."""

import argparse
import logging
from pathlib import Path
import sys
from typing import List, Tuple
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import BASELINE_CONFIG, TRANSFORMER_CONFIG
from preprocess import clean_text
from src.baseline import BaselineToxicityClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("predict")


def load_classifier(model_type: str = "baseline"):
    """Load requested model for inference."""
    if model_type == "baseline":
        if not BASELINE_CONFIG.model_path.exists():
            raise FileNotFoundError(
                f"Baseline model not found at {BASELINE_CONFIG.model_path}. Train it first: 'python train.py --model baseline'"
            )
        return BaselineToxicityClassifier.load(BASELINE_CONFIG.model_path)
    
    elif model_type == "transformer":
        if not TRANSFORMER_CONFIG.model_dir.exists():
            raise FileNotFoundError(
                f"Transformer model directory not found at {TRANSFORMER_CONFIG.model_dir}. "
                "Train it first or download fine-tuned weights."
            )
        from src.transformer_model import XLMRobertaToxicityClassifier
        model = XLMRobertaToxicityClassifier(model_name_or_path=TRANSFORMER_CONFIG.model_dir)
        model.load()
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def predict_single(text: str, model_type: str = "baseline", threshold: float = 0.5):
    """Run prediction on a single input string."""
    cleaned = clean_text(text)
    if not cleaned:
        print("\nInput text was empty or contained only whitespace/control characters.")
        return

    model = load_classifier(model_type)
    probs = model.predict_proba([cleaned])[0]
    p_non_toxic, p_toxic = float(probs[0]), float(probs[1])
    is_toxic = p_toxic >= threshold

    verdict = "⚠️ TOXIC" if is_toxic else "✅ NON-TOXIC"
    confidence = max(p_non_toxic, p_toxic) * 100

    print("\n" + "=" * 60)
    print(f"MODEL PREDICTION ({model_type.upper()})")
    print("=" * 60)
    print(f"Raw Input:       {text}")
    print(f"Cleaned Text:    {cleaned}")
    print(f"Verdict:         {verdict}")
    print(f"Toxic Prob:      {p_toxic:.4f} ({p_toxic*100:.2f}%)")
    print(f"Non-Toxic Prob:  {p_non_toxic:.4f} ({p_non_toxic*100:.2f}%)")
    print(f"Threshold:       {threshold:.2f}")
    print(f"Confidence:      {confidence:.1f}%")
    print("=" * 60 + "\n")


def predict_batch(file_path: str, output_path: str, text_col: str = "comment_text", model_type: str = "baseline", threshold: float = 0.5):
    """Run prediction on batch CSV file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Batch file not found: {path}")

    df = pd.read_csv(path)
    if text_col not in df.columns:
        # Fallback search
        for c in ["text", "comment", "clean_text"]:
            if c in df.columns:
                text_col = c
                break
        else:
            raise ValueError(f"Column '{text_col}' not found in CSV. Available: {list(df.columns)}")

    logger.info(f"Processing {len(df)} comments from {path} using {model_type}...")
    cleaned_texts = [clean_text(t) for t in df[text_col]]
    
    model = load_classifier(model_type)
    probs = model.predict_proba(cleaned_texts)
    toxic_probs = probs[:, 1]
    predictions = (toxic_probs >= threshold).astype(int)

    df["toxicity_prob"] = toxic_probs
    df["predicted_label"] = predictions
    df["verdict"] = np.where(predictions == 1, "Toxic", "Non-toxic")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Predictions successfully written to: {out}")
    print(f"Summary: {int(np.sum(predictions == 1))} Toxic / {int(np.sum(predictions == 0))} Non-toxic out of {len(df)} total.")


def parse_args():
    parser = argparse.ArgumentParser(description="Multilingual Toxic Comment Prediction CLI")
    parser.add_argument("--text", type=str, default=None, help="Single comment text string to evaluate")
    parser.add_argument("--file", type=str, default=None, help="Input CSV path for batch inference")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV path for batch predictions")
    parser.add_argument("--text-col", type=str, default="comment_text", help="Column name containing text in batch CSV")
    parser.add_argument("--model", type=str, choices=["baseline", "transformer"], default="baseline", help="Model to use")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for toxicity classification")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.text is not None:
        predict_single(args.text, model_type=args.model, threshold=args.threshold)
    elif args.file is not None:
        predict_batch(args.file, args.output, text_col=args.text_col, model_type=args.model, threshold=args.threshold)
    else:
        print("Please provide either --text <string> or --file <csv_path>. Use --help for options.")


if __name__ == "__main__":
    main()
