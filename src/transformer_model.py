"""XLM-RoBERTa fine-tuning and inference implementation for Multilingual Toxic Comment Classification."""

import json
import logging
import math
import os
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from config import TRANSFORMER_CONFIG, get_default_device
from src.dataset import ToxicityDataset

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class XLMRobertaToxicityClassifier:
    """Inference and wrapper interface for the fine-tuned XLM-RoBERTa model.
    
    Academic Rationale:
    XLM-RoBERTa (Conneau et al., 2020) is pre-trained on 2.5TB of filtered CommonCrawl data
    covering 100 languages using masked language modeling. Its 250k subword vocabulary
    allows shared cross-lingual representations, enabling cross-lingual zero-shot transfer
    from high-resource languages (English) to lower-resource languages.
    """

    def __init__(
        self,
        model_name_or_path: Union[str, Path] = "FacebookAI/xlm-roberta-base",
        device: Optional[torch.device] = None,
        max_length: int = 128,
        default_threshold: float = 0.5,
    ):
        self.model_name_or_path = str(model_name_or_path)
        self.device = device or get_default_device()
        self.max_length = max_length
        self.default_threshold = default_threshold

        self.tokenizer = None
        self.model = None
        self._is_loaded = False

    def load(self) -> "XLMRobertaToxicityClassifier":
        """Load tokenizer and model weights onto target hardware."""
        logger.info(f"Loading XLM-RoBERTa from {self.model_name_or_path} onto {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                num_labels=2,
            )
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            logger.info("XLM-RoBERTa loaded successfully.")
            return self
        except Exception as e:
            logger.error(f"Failed to load XLM-RoBERTa model: {e}")
            raise RuntimeError(
                f"Could not load XLM-RoBERTa from '{self.model_name_or_path}'. "
                "Ensure model weights exist at path or active internet connection is available."
            ) from e

    def predict_proba(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Compute predicted class probabilities [P(Non-toxic), P(Toxic)] for input texts."""
        if not self._is_loaded:
            self.load()

        all_probs = []
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)

        dataset = ToxicityDataset(texts=texts, tokenizer=self.tokenizer, max_length=self.max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def predict(self, texts: List[str], threshold: Optional[float] = None) -> np.ndarray:
        """Predict binary labels using decision threshold."""
        thresh = threshold if threshold is not None else self.default_threshold
        probs = self.predict_proba(texts)
        toxic_probs = probs[:, 1]
        return (toxic_probs >= thresh).astype(int)


def train_transformer_model(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    output_dir: Union[str, Path] = TRANSFORMER_CONFIG.model_dir,
    pretrained_model_name: str = TRANSFORMER_CONFIG.pretrained_model_name,
    learning_rate: float = TRANSFORMER_CONFIG.learning_rate,
    epochs: int = TRANSFORMER_CONFIG.epochs,
    batch_size: int = TRANSFORMER_CONFIG.batch_size,
    eval_batch_size: int = TRANSFORMER_CONFIG.eval_batch_size,
    max_length: int = TRANSFORMER_CONFIG.max_length,
    weight_decay: float = TRANSFORMER_CONFIG.weight_decay,
    warmup_ratio: float = TRANSFORMER_CONFIG.warmup_ratio,
    seed: int = TRANSFORMER_CONFIG.random_seed,
    fp16: Optional[bool] = None,
    handle_class_imbalance: bool = True,
) -> Dict[str, Union[float, str]]:
    """Fine-tune XLM-RoBERTa for toxicity classification with dynamic padding and validation checkpointing.
    
    Args:
        train_texts: Training comments.
        train_labels: Binary training labels.
        val_texts: Validation comments.
        val_labels: Binary validation labels.
        output_dir: Path to directory where best weights and tokenizer are saved.
        pretrained_model_name: Hugging Face model identifier.
        learning_rate: Peak AdamW learning rate.
        epochs: Number of training epochs.
        batch_size: Training batch size per device.
        eval_batch_size: Evaluation batch size.
        max_length: Maximum token length for truncation.
        weight_decay: Weight decay for regularized optimization.
        warmup_ratio: Warmup steps fraction.
        seed: Random reproducibility seed.
        fp16: Enable mixed-precision training on GPU.
        handle_class_imbalance: Weight loss inversely by class frequency.
        
    Returns:
        Dictionary containing final evaluation metrics and training metadata.
    """
    set_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    device = get_default_device()
    use_fp16 = (fp16 if fp16 is not None else torch.cuda.is_available()) and (device.type == "cuda")

    logger.info(f"Initiating XLM-RoBERTa training on device: {device} (FP16={use_fp16})")
    logger.info(f"Dataset stats: Train={len(train_texts)}, Val={len(val_texts)}")

    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name,
        num_labels=2,
    )
    model.to(device)

    # Class weighting for loss function if justified
    loss_fct = nn.CrossEntropyLoss()
    if handle_class_imbalance:
        class_counts = np.bincount(train_labels)
        total_samples = len(train_labels)
        # Inverse class frequency weighting
        weights = [total_samples / (2.0 * count) for count in class_counts]
        weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
        loss_fct = nn.CrossEntropyLoss(weight=weights_tensor)
        logger.info(f"Class imbalance weighting applied: {weights}")

    # Datasets and Dynamic Padding Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    train_dataset = ToxicityDataset(train_texts, train_labels, tokenizer=tokenizer, max_length=max_length)
    val_dataset = ToxicityDataset(val_texts, val_labels, tokenizer=tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    # Optimizer & LR Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    best_val_f1 = -1.0
    best_metrics = {}
    history = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        logger.info(f"--- Epoch {epoch}/{epochs} ---")

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            if use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fct(outputs.logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fct(outputs.logits, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()
            total_train_loss += loss.item()

            if (step + 1) % max(1, len(train_loader) // 5) == 0:
                logger.info(f"Step {step+1}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.numpy())

        # Calculate epoch validation F1
        val_targets = np.array(val_targets)
        val_preds = np.array(val_preds)
        tp = np.sum((val_targets == 1) & (val_preds == 1))
        fp = np.sum((val_targets == 0) & (val_preds == 1))
        fn = np.sum((val_targets == 1) & (val_preds == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        val_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        val_acc = np.mean(val_preds == val_targets)

        logger.info(
            f"Epoch {epoch} Results: Train Loss={avg_train_loss:.4f} | "
            f"Val Acc={val_acc:.4f} | Val Toxic F1={val_f1:.4f} (P={precision:.4f}, R={recall:.4f})"
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_accuracy": float(val_acc),
            "val_f1": float(val_f1),
        })

        # Save Best Model Checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_metrics = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_accuracy": float(val_acc),
                "val_f1": float(val_f1),
                "precision": float(precision),
                "recall": float(recall),
            }
            logger.info(f"Validation F1 improved to {best_val_f1:.4f}. Saving best model checkpoint to {output_path}...")
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.1f} seconds. Best Val F1: {best_val_f1:.4f}")

    # Save training configuration & metrics summary
    metadata = {
        "model_name": pretrained_model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "device": str(device),
        "fp16": use_fp16,
        "best_metrics": best_metrics,
        "history": history,
        "training_time_sec": training_time,
    }
    with open(output_path / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata
