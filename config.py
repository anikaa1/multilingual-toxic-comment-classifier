"""Configuration settings for the Multilingual Toxic Comment Classifier.

Centralizes paths, hyperparameters, column mappings, and hardware detection.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional, Tuple

# Suppress font cache warning in headless / restricted environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch


# Project Root Directory
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_default_device() -> torch.device:
    """Detect the most optimal hardware acceleration available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class DatasetConfig:
    """Dataset file locations and column configurations."""
    train_file: Path = DATA_DIR / "jigsaw-toxic-comment-train.csv"
    val_file: Path = DATA_DIR / "validation.csv"
    test_file: Path = DATA_DIR / "test.csv"
    synthetic_file: Path = DATA_DIR / "synthetic_smoke_test.csv"

    # Column name mappings with fallbacks
    text_col: str = "comment_text"
    label_col: str = "toxic"
    lang_col: str = "lang"
    id_col: str = "id"

    # Data split ratios (for single dataset partitioning)
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    random_seed: int = 42


@dataclass
class BaselineConfig:
    """TF-IDF and Logistic Regression baseline hyperparameters."""
    # Word n-gram TF-IDF
    word_ngram_range: Tuple[int, int] = (1, 2)
    word_max_features: int = 15000
    
    # Character n-gram TF-IDF (crucial for typos, obfuscation, subwords)
    char_ngram_range: Tuple[int, int] = (2, 5)
    char_max_features: int = 25000
    
    # Logistic Regression
    C: float = 2.0
    max_iter: int = 1000
    class_weight: Optional[str] = "balanced"
    solver: str = "lbfgs"
    random_seed: int = 42

    # Artifact paths
    model_path: Path = MODELS_DIR / "baseline_model.joblib"


@dataclass
class TransformerConfig:
    """XLM-RoBERTa fine-tuning hyperparameters."""
    pretrained_model_name: str = "FacebookAI/xlm-roberta-base"
    num_labels: int = 2
    max_length: int = 128
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    epochs: int = 3
    gradient_accumulation_steps: int = 1
    fp16: bool = torch.cuda.is_available()  # Enable fp16 if CUDA is available
    random_seed: int = 42
    default_threshold: float = 0.5

    # Directory for saving checkpoints and final weights
    model_dir: Path = MODELS_DIR / "xlm-roberta-toxic"


@dataclass
class AppConfig:
    """Streamlit application settings."""
    app_title: str = "Multilingual Toxic Comment Classifier"
    default_threshold: float = 0.50
    baseline_model_path: Path = MODELS_DIR / "baseline_model.joblib"
    transformer_model_dir: Path = MODELS_DIR / "xlm-roberta-toxic"
    results_dir: Path = RESULTS_DIR


# Instantiated singleton defaults
DATASET_CONFIG = DatasetConfig()
BASELINE_CONFIG = BaselineConfig()
TRANSFORMER_CONFIG = TransformerConfig()
APP_CONFIG = AppConfig()
