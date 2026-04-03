"""
model/trainer.py
Full training loop for PhishGuard DistilBERT classifier.
Features: mixed precision, early stopping, gradient clipping, checkpointing.
"""

import pathlib
import yaml
import logging
import time
import json
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Dict, Optional, List
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from model.distilbert_classifier import PhishingClassifier, save_model

ROOT = pathlib.Path(__file__).parent.parent
CONFIG = yaml.safe_load(open(ROOT / "config.yaml"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / "training.log", mode="a"),
    ]
)
logger = logging.getLogger(__name__)

(ROOT / "logs").mkdir(exist_ok=True)


class EarlyStopping:
    """Stop training when val F1 doesn't improve for `patience` epochs."""

    def __init__(self, patience: int = 2, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_f1: float) -> bool:
        if self.best_score is None:
            self.best_score = val_f1
        elif val_f1 < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = val_f1
            self.counter = 0
        return self.should_stop


def evaluate(
    model: PhishingClassifier,
    loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Run evaluation and return metrics."""
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            loss, logits = model(input_ids, attention_mask, labels)
            preds = torch.argmax(logits, dim=-1)

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average="binary")
    precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    return {
        "loss": round(avg_loss, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4),
    }


def train(
    model: PhishingClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    config: Optional[Dict] = None,
) -> Dict[str, List]:
    """
    Full training loop with:
    - Mixed precision training (AMP)
    - Linear warmup LR scheduler
    - Gradient clipping
    - Early stopping
    - Best model checkpointing
    - Step-level logging

    Returns:
        Training history dict
    """
    config = config or CONFIG["training"]
    lr = config["learning_rate"]
    epochs = config["epochs"]
    warmup_ratio = config["warmup_ratio"]
    grad_clip = config["gradient_clip"]
    log_every = config["log_every_steps"]
    patience = config["early_stopping_patience"]
    use_amp = config["mixed_precision"] and device == "cuda"

    model_save_path = ROOT / CONFIG["paths"]["model"]
    model_save_path.mkdir(parents=True, exist_ok=True)

    # Optimizer — separate LR for backbone vs classifier head
    optimizer = AdamW([
        {"params": model.distilbert.parameters(), "lr": lr},
        {"params": model.pre_classifier.parameters(), "lr": lr * 5},
        {"params": model.classifier.parameters(), "lr": lr * 5},
    ], weight_decay=config["weight_decay"])

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = GradScaler() if use_amp else None
    early_stopping = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_precision": [], "val_recall": []}
    best_val_f1 = 0.0
    global_step = 0

    logger.info(f"Starting training: {epochs} epochs | {len(train_loader)} steps/epoch | device={device}")
    logger.info(f"Mixed precision: {use_amp} | Warmup steps: {warmup_steps}")
    logger.info("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_losses = []
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    loss, _ = model(input_ids, attention_mask, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, _ = model(input_ids, attention_mask, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            scheduler.step()
            global_step += 1
            step_loss = loss.item()
            epoch_loss += step_loss
            step_losses.append(step_loss)

            # Log every N steps
            if global_step % log_every == 0:
                avg_step_loss = sum(step_losses[-log_every:]) / min(len(step_losses), log_every)
                elapsed = time.time() - t0
                steps_per_sec = step / elapsed
                logger.info(
                    f"Epoch {epoch}/{epochs} | Step {global_step} | "
                    f"Loss: {avg_step_loss:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"{steps_per_sec:.1f} steps/s"
                )

        # End of epoch — evaluate
        avg_train_loss = epoch_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])

        epoch_time = time.time() - t0
        logger.info("=" * 60)
        logger.info(
            f"EPOCH {epoch}/{epochs} COMPLETE ({epoch_time:.1f}s)\n"
            f"  Train Loss:  {avg_train_loss:.4f}\n"
            f"  Val Loss:    {val_metrics['loss']:.4f}\n"
            f"  Val F1:      {val_metrics['f1']:.4f}\n"
            f"  Val Prec:    {val_metrics['precision']:.4f}\n"
            f"  Val Recall:  {val_metrics['recall']:.4f}\n"
            f"  Val Acc:     {val_metrics['accuracy']:.4f}"
        )
        logger.info("=" * 60)

        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            save_model(model, model_save_path)
            logger.info(f"✅ New best model saved! Val F1: {best_val_f1:.4f}")

        # Early stopping
        if early_stopping(val_metrics["f1"]):
            logger.info(f"⏹ Early stopping triggered after epoch {epoch}")
            break

    # Save training history
    history_path = ROOT / "logs" / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\n🎉 Training complete! Best Val F1: {best_val_f1:.4f}")
    logger.info(f"Model saved to: {model_save_path}")
    return history
