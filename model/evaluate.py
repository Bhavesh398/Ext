"""
model/evaluate.py
Comprehensive model evaluation with metrics, plots, and error analysis.
"""

import pathlib
import yaml
import json
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from model.distilbert_classifier import PhishingClassifier

ROOT = pathlib.Path(__file__).parent.parent
CONFIG = yaml.safe_load(open(ROOT / "config.yaml"))
logger = logging.getLogger(__name__)

TARGETS = {
    "f1": 0.96,
    "recall": 0.97,
    "precision": 0.95,
    "roc_auc": 0.98,
}


def get_predictions(
    model: PhishingClassifier,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model on all batches and collect predictions.

    Returns:
        (all_labels, all_preds, all_probs)
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].numpy()

            _, logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_labels.extend(labels)
            all_preds.extend(preds)
            all_probs.extend(probs)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    return {
        "f1":        round(f1_score(labels, preds, average="binary"), 4),
        "precision": round(precision_score(labels, preds, average="binary", zero_division=0), 4),
        "recall":    round(recall_score(labels, preds, average="binary", zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(labels, probs), 4),
        "accuracy":  round(float((preds == labels).mean()), 4),
    }


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    save_dir: pathlib.Path,
):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)

    classes = ["Safe", "Phishing"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title("Confusion Matrix — PhishGuard", fontsize=14, fontweight="bold")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = save_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved: {path}")


def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    save_dir: pathlib.Path,
):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#3b82f6", lw=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#3b82f6")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — PhishGuard", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = save_dir / "roc_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curve saved: {path}")


def plot_score_distribution(
    labels: np.ndarray,
    probs: np.ndarray,
    save_dir: pathlib.Path,
):
    """Plot phishing score distribution for phishing vs safe emails."""
    scores = (probs * 100).astype(int)
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(scores[labels == 0], bins=50, alpha=0.6, color="#10b981",
            label="Safe emails", density=True)
    ax.hist(scores[labels == 1], bins=50, alpha=0.6, color="#ef4444",
            label="Phishing emails", density=True)

    ax.axvline(x=40, color="orange", linestyle="--", lw=1.5, label="Suspicious threshold (40)")
    ax.axvline(x=70, color="red", linestyle="--", lw=1.5, label="Phishing threshold (70)")

    ax.set_xlabel("Phishing Score (0-100)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Score Distribution — PhishGuard", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = save_dir / "score_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Score distribution saved: {path}")


def check_targets(metrics: Dict[str, float]) -> bool:
    """Check if model meets production targets."""
    logger.info("\n── Target Check ───────────────────────────────────────")
    all_pass = True
    for metric, target in TARGETS.items():
        value = metrics.get(metric, 0)
        status = "✅" if value >= target else "❌"
        logger.info(f"  {status} {metric}: {value:.4f} (target: ≥{target})")
        if value < target:
            all_pass = False
    logger.info("──────────────────────────────────────────────────────")
    return all_pass


def evaluate_model(
    model: PhishingClassifier,
    test_loader: DataLoader,
    device: str,
    save_plots: bool = True,
) -> Dict[str, float]:
    """
    Full evaluation pipeline.

    Args:
        model: Trained PhishingClassifier
        test_loader: Test DataLoader
        device: 'cuda' or 'cpu'
        save_plots: Whether to save metric plots

    Returns:
        Dict of evaluation metrics
    """
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    logger.info("Running evaluation on test set...")
    labels, preds, probs = get_predictions(model, test_loader, device)
    metrics = compute_metrics(labels, preds, probs)

    logger.info("\n══════════════════════════════════════════════════════")
    logger.info("             PHISHGUARD EVALUATION RESULTS")
    logger.info("══════════════════════════════════════════════════════")
    for k, v in metrics.items():
        logger.info(f"  {k.upper():<15} {v:.4f}")
    logger.info("══════════════════════════════════════════════════════")
    logger.info("\nClassification Report:")
    logger.info(classification_report(labels, preds, target_names=["Safe", "Phishing"]))

    passed = check_targets(metrics)
    if not passed:
        logger.warning("⚠️  Model does not meet production targets. Consider:")
        logger.warning("   - Training for more epochs")
        logger.warning("   - Reducing learning rate to 1e-5")
        logger.warning("   - Adding more phishing training samples")

    if save_plots:
        plot_confusion_matrix(labels, preds, logs_dir)
        plot_roc_curve(labels, probs, logs_dir)
        plot_score_distribution(labels, probs, logs_dir)

    # Save metrics to JSON
    metrics["targets_passed"] = passed
    with open(logs_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
