"""Multilingual Toxic Comment Classifier package source.

Contains dataset handling, baseline TF-IDF model, fine-tuned XLM-RoBERTa model,
and comprehensive academic evaluation metrics.
"""

from src.dataset import ToxicityDataset, load_and_prepare_datasets
from src.baseline import BaselineToxicityClassifier
from src.transformer_model import XLMRobertaToxicityClassifier, train_transformer_model
from src.metrics import compute_classification_metrics, evaluate_model_performance

__all__ = [
    "ToxicityDataset",
    "load_and_prepare_datasets",
    "BaselineToxicityClassifier",
    "XLMRobertaToxicityClassifier",
    "train_transformer_model",
    "compute_classification_metrics",
    "evaluate_model_performance",
]
