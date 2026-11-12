"""Data preprocessing and text normalization pipeline.

Designed specifically for toxicity classification:
- Preserves casing, emotional punctuation, and emojis (critical toxicity signals).
- Normalizes structural noise (HTML tags, URLs, user mentions).
- Handles duplicates, missing values, and data leakage across splits.
"""

import html
import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# Regex patterns for normalization
URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
USER_MENTION_REGEX = re.compile(r"@[\w\-]+", re.IGNORECASE)
HTML_TAG_REGEX = re.compile(r"<[^>]+>")
WHITESPACE_REGEX = re.compile(r"[ \t]+")
EXCESSIVE_NEWLINES_REGEX = re.compile(r"\n\s*\n+")


def clean_text(text: Union[str, float, None], normalize_urls: bool = True, normalize_mentions: bool = True) -> str:
    """Clean a single raw text string while preserving crucial toxicity markers.
    
    Academic Rationale:
    1. Casing (e.g., ALL CAPS) carries strong emotional intensity and hostility.
    2. Punctuation (e.g., '!!!', '???') indicates aggression and urgency.
    3. Emojis carry explicit sentiment and toxicity signals.
    4. Subword tokenizers (Byte-Pair / SentencePiece) in models like XLM-RoBERTa
       rely on true casing and punctuation to disambiguate meaning.
       
    Therefore, we avoid aggressive lowercasing, stopword stripping, or stemming.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    
    text = str(text)

    # 1. Unescape HTML entities (e.g., &amp; -> &, &lt; -> <)
    text = html.unescape(text)

    # 2. Strip HTML tags
    text = HTML_TAG_REGEX.sub(" ", text)

    # 3. Normalize URLs to a uniform token to prevent vocabulary explosion
    if normalize_urls:
        text = URL_REGEX.sub("[URL]", text)

    # 4. Normalize user mentions
    if normalize_mentions:
        text = USER_MENTION_REGEX.sub("[USER]", text)

    # 5. Remove null bytes and non-printable control characters (keep standard newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 6. Normalize excessive horizontal whitespace and repeated blank lines
    text = EXCESSIVE_NEWLINES_REGEX.sub("\n", text)
    text = WHITESPACE_REGEX.sub(" ", text)

    return text.strip()


def prepare_dataframe(
    df: pd.DataFrame,
    text_col: str = "comment_text",
    label_col: Optional[str] = "toxic",
    lang_col: Optional[str] = "lang",
    drop_duplicates: bool = True,
    binary_threshold: float = 0.5,
) -> pd.DataFrame:
    """Preprocess, sanitize, and validate a toxicity dataset DataFrame.
    
    Args:
        df: Input pandas DataFrame.
        text_col: Name of column containing comments.
        label_col: Name of column containing toxicity target.
        lang_col: Name of column indicating comment language code.
        drop_duplicates: Whether to drop duplicate comments.
        binary_threshold: Threshold to convert continuous ratings (0.0-1.0) to binary (0 or 1).
        
    Returns:
        Cleaned pandas DataFrame with normalized columns.
    """
    initial_rows = len(df)
    logger.info(f"Preparing dataset with initial {initial_rows} rows.")

    # Resolve text column fallback
    resolved_text_col = None
    for candidate in [text_col, "text", "comment_text", "comment", "content"]:
        if candidate in df.columns:
            resolved_text_col = candidate
            break

    if resolved_text_col is None:
        raise ValueError(f"Could not find text column. Looked for '{text_col}', 'text', 'comment_text' in columns: {list(df.columns)}")

    df = df.copy()
    # Normalize text column
    df["clean_text"] = df[resolved_text_col].apply(clean_text)

    # Remove empty or whitespace-only texts
    df = df[df["clean_text"].str.len() > 0]
    non_empty_rows = len(df)
    logger.info(f"Filtered {initial_rows - non_empty_rows} empty/null rows.")

    # Deduplication
    if drop_duplicates:
        df = df.drop_duplicates(subset=["clean_text"])
        logger.info(f"Removed {non_empty_rows - len(df)} duplicate comments. Remaining: {len(df)}")

    # Process target label if present
    if label_col is not None:
        resolved_label_col = None
        for candidate in [label_col, "label", "toxic", "toxicity", "target"]:
            if candidate in df.columns:
                resolved_label_col = candidate
                break

        if resolved_label_col is not None:
            # Handle continuous target values (e.g. Civil Comments target score between 0 and 1)
            raw_labels = df[resolved_label_col].astype(float)
            df["label"] = (raw_labels >= binary_threshold).astype(int)
            class_counts = df["label"].value_counts().to_dict()
            logger.info(f"Target distribution (0=Non-toxic, 1=Toxic): {class_counts}")
        else:
            logger.warning(f"Label column '{label_col}' not found. Dataset will have no 'label' column.")

    # Preserve language column if available
    if lang_col and lang_col in df.columns:
        df["lang"] = df[lang_col].fillna("unknown").astype(str).str.lower()
    elif "lang" not in df.columns:
        df["lang"] = "unknown"

    df = df.reset_index(drop=True)
    return df


def verify_no_data_leakage(train_texts: List[str], eval_texts: List[str], split_name: str = "val") -> int:
    """Verify that no identical comments appear in both training and evaluation splits."""
    train_set: Set[str] = set(train_texts)
    eval_set: Set[str] = set(eval_texts)
    leakage = train_set.intersection(eval_set)
    if leakage:
        logger.warning(
            f"DATA LEAKAGE WARNING: Found {len(leakage)} overlapping samples between train and {split_name} split! "
            "These must be excluded from evaluation."
        )
    else:
        logger.info(f"Data leakage check passed: 0 overlapping samples between train and {split_name}.")
    return len(leakage)


def split_dataset(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    label_col: str = "label",
    train_size: float = 0.80,
    val_size: float = 0.10,
    test_size: float = 0.10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform stratified split into Train, Validation, and Test partitions with leakage verification.
    
    Args:
        df: Preprocessed DataFrame containing clean text and labels.
        text_col: Text column name.
        label_col: Label column name.
        train_size: Proportion of training data (e.g. 0.8).
        val_size: Proportion of validation data (e.g. 0.1).
        test_size: Proportion of test data (e.g. 0.1).
        random_state: Reproducibility seed.
        
    Returns:
        (train_df, val_df, test_df)
    """
    total = train_size + val_size + test_size
    assert abs(total - 1.0) < 1e-5, f"Split proportions must sum to 1.0, got {total}"

    has_labels = label_col in df.columns
    stratify = df[label_col] if has_labels else None

    # First split off the train partition
    train_df, remaining_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Next split the remainder into validation and test partitions
    remaining_ratio = val_size + test_size
    val_relative_ratio = val_size / remaining_ratio
    remaining_stratify = remaining_df[label_col] if has_labels else None

    val_df, test_df = train_test_split(
        remaining_df,
        train_size=val_relative_ratio,
        random_state=random_state,
        stratify=remaining_stratify,
    )

    # Leakage check and cleanup
    train_texts = set(train_df[text_col].tolist())
    val_leakage = val_df[val_df[text_col].isin(train_texts)]
    test_leakage = test_df[test_df[text_col].isin(train_texts)]

    if len(val_leakage) > 0:
        logger.warning(f"Removing {len(val_leakage)} leaked rows from validation set.")
        val_df = val_df[~val_df[text_col].isin(train_texts)]

    if len(test_leakage) > 0:
        logger.warning(f"Removing {len(test_leakage)} leaked rows from test set.")
        test_df = test_df[~test_df[text_col].isin(train_texts)]

    logger.info(
        f"Split complete: Train={len(train_df)} ({len(train_df)/len(df):.1%}), "
        f"Val={len(val_df)} ({len(val_df)/len(df):.1%}), "
        f"Test={len(test_df)} ({len(test_df)/len(df):.1%})"
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
