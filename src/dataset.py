"""PyTorch Dataset implementation and dataset loading utilities for toxicity classification."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocess import prepare_dataframe, split_dataset
from config import DATASET_CONFIG

logger = logging.getLogger(__name__)


class ToxicityDataset(Dataset):
    """PyTorch Dataset for text sequence classification.
    
    Supports Hugging Face tokenizers with dynamic truncation and conversion
    to PyTorch tensors.
    """

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer=None,
        max_length: int = 128,
    ):
        """
        Args:
            texts: List of preprocessed text strings.
            labels: Optional list of binary labels (0 or 1).
            tokenizer: Hugging Face PreTrainedTokenizer or PreTrainedTokenizerFast.
            max_length: Maximum sequence length (tokens).
        """
        self.texts = [str(t) for t in texts]
        self.labels = [int(lbl) for lbl in labels] if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False,  # Defer padding to DataCollatorWithPadding for efficiency
                return_tensors=None,
            )
            item = {
                "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            }
        else:
            # Fallback if raw text item needed
            item = {"text": text}

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def load_and_prepare_datasets(
    train_path: Optional[Union[str, Path]] = None,
    val_path: Optional[Union[str, Path]] = None,
    test_path: Optional[Union[str, Path]] = None,
    use_synthetic: bool = False,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, clean, and split training, validation, and test datasets.
    
    Supports:
    1. Synthetic mode: loads small multi-lingual synthetic dataset for pipeline verification.
    2. Competition mode: loads separate train, validation, and test files if present.
    3. Single file mode: if only train_path exists, performs a stratified train/val/test split.
    
    Args:
        train_path: Path to raw training CSV.
        val_path: Optional path to validation CSV.
        test_path: Optional path to test CSV.
        use_synthetic: If True, forces use of synthetic smoke-test data.
        sample_size: Optional row limit for fast local testing on a subset of full data.
        random_seed: Reproducibility seed.
        
    Returns:
        (train_df, val_df, test_df) preprocessed and sanitized DataFrames.
    """
    if use_synthetic:
        synthetic_path = DATASET_CONFIG.synthetic_file
        if not synthetic_path.exists():
            raise FileNotFoundError(f"Synthetic dataset file not found at {synthetic_path}")
        logger.info(f"Loading synthetic smoke-test dataset from {synthetic_path}")
        df_raw = pd.read_csv(synthetic_path)
        df_clean = prepare_dataframe(
            df_raw,
            text_col="comment_text",
            label_col="toxic",
            lang_col="lang",
            drop_duplicates=True,
        )
        return split_dataset(
            df_clean,
            train_size=0.60,
            val_size=0.20,
            test_size=0.20,
            random_state=random_seed,
        )

    # Determine paths
    train_file = Path(train_path) if train_path else DATASET_CONFIG.train_file
    val_file = Path(val_path) if val_path else DATASET_CONFIG.val_file
    test_file = Path(test_path) if test_path else DATASET_CONFIG.test_file

    if not train_file.exists():
        raise FileNotFoundError(
            f"Training data file not found at: {train_file}\n"
            "Please follow data/download_instructions.md to obtain the Jigsaw dataset,\n"
            "or use the '--use-synthetic' flag for verification."
        )

    logger.info(f"Loading training data from {train_file}")
    # Load training file
    train_raw = pd.read_csv(train_file, nrows=sample_size)
    train_df = prepare_dataframe(
        train_raw,
        text_col=DATASET_CONFIG.text_col,
        label_col=DATASET_CONFIG.label_col,
        lang_col=DATASET_CONFIG.lang_col,
    )

    # If separate validation and test files exist (e.g. Jigsaw competition layout)
    if val_file.exists() and test_file.exists():
        logger.info(f"Loading separate validation set from {val_file}")
        val_raw = pd.read_csv(val_file, nrows=sample_size)
        val_df = prepare_dataframe(
            val_raw,
            text_col=DATASET_CONFIG.text_col,
            label_col=DATASET_CONFIG.label_col,
            lang_col=DATASET_CONFIG.lang_col,
        )

        logger.info(f"Loading separate test set from {test_file}")
        test_raw = pd.read_csv(test_file, nrows=sample_size)
        test_df = prepare_dataframe(
            test_raw,
            text_col=DATASET_CONFIG.text_col,
            label_col=DATASET_CONFIG.label_col,
            lang_col=DATASET_CONFIG.lang_col,
        )
        return train_df, val_df, test_df
    
    # Otherwise, split the single training dataset into train, val, and test
    logger.info("Separate validation/test files not found. Creating stratified train/val/test splits.")
    return split_dataset(
        train_df,
        text_col="clean_text",
        label_col="label",
        train_size=DATASET_CONFIG.train_ratio,
        val_size=DATASET_CONFIG.val_ratio,
        test_size=DATASET_CONFIG.test_ratio,
        random_state=random_seed,
    )
