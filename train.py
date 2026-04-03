"""
train.py
Main entry point to run the full PhishGuard ML training pipeline.

Usage:
    python train.py                    # Full pipeline
    python train.py --skip-data        # Skip data prep if already done
    python train.py --epochs 3         # Override epochs
    python train.py --eval-only        # Only run evaluation on saved model
"""

import argparse
import pathlib
import logging
import yaml
import torch
import sys

ROOT = pathlib.Path(__file__).parent
sys.path.append(str(ROOT))

(ROOT / "logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / "train.log", mode="w", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

CONFIG = yaml.safe_load(open(ROOT / "config.yaml"))


def parse_args():
    parser = argparse.ArgumentParser(description="PhishGuard DistilBERT Training")
    parser.add_argument("--skip-data", action="store_true", help="Skip data preparation")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate saved model")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    return parser.parse_args()


def main():
    args = parse_args()

    # Override config if args provided
    if args.epochs:
        CONFIG["training"]["epochs"] = args.epochs
    if args.batch_size:
        CONFIG["training"]["batch_size"] = args.batch_size
    if args.lr:
        CONFIG["training"]["learning_rate"] = args.lr

    # ── Device setup ──────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🚀 GPU detected: {gpu_name} ({gpu_mem:.1f}GB VRAM)")

        # Auto-adjust batch size based on VRAM
        if gpu_mem < 6:
            CONFIG["training"]["batch_size"] = 8
            logger.info("⚠️  Low VRAM detected, reducing batch size to 8")
        elif gpu_mem >= 16:
            CONFIG["training"]["batch_size"] = 32
            logger.info("💪 High VRAM detected, increasing batch size to 32")
    else:
        logger.warning("⚠️  No GPU found, using CPU. Training will be slow!")
        CONFIG["training"]["batch_size"] = 4
        CONFIG["training"]["epochs"] = min(CONFIG["training"]["epochs"], 2)

    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {CONFIG['training']['epochs']}")
    logger.info(f"Batch size: {CONFIG['training']['batch_size']}")
    logger.info(f"Learning rate: {CONFIG['training']['learning_rate']}")

    # ── Step 1: Data preparation ──────────────────────────────────────────
    if not args.skip_data and not args.eval_only:
        logger.info("\n📦 STEP 1: Preparing data...")
        from data.data_loader import prepare_data
        train_df, val_df, test_df = prepare_data()
        logger.info(f"Data ready: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")
    else:
        logger.info("⏭  Skipping data prep, loading cached splits...")
        import pandas as pd
        processed_dir = ROOT / CONFIG["paths"]["processed_data"]
        train_df = pd.read_csv(processed_dir / "train.csv")
        val_df = pd.read_csv(processed_dir / "val.csv")
        test_df = pd.read_csv(processed_dir / "test.csv")

    # ── Step 2: Tokenizer + DataLoaders ──────────────────────────────────
    logger.info("\n🔤 STEP 2: Setting up tokenizer and data loaders...")
    from model.dataset import get_tokenizer, get_dataloaders
    tokenizer = get_tokenizer()
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, tokenizer,
        batch_size=CONFIG["training"]["batch_size"]
    )

    # ── Step 3: Model setup ───────────────────────────────────────────────
    logger.info("\n🧠 STEP 3: Initializing DistilBERT classifier...")
    from model.distilbert_classifier import PhishingClassifier, load_model
    from model.evaluate import evaluate_model

    model_path = ROOT / CONFIG["paths"]["model"]

    if args.eval_only:
        logger.info("Loading saved model for evaluation only...")
        model = load_model(model_path, tokenizer_vocab_size=len(tokenizer), device=device)
    else:
        model = PhishingClassifier(
            vocab_size=len(tokenizer),
            dropout=CONFIG["model"]["dropout"],
            hidden_size=CONFIG["model"]["hidden_size"],
        )
        model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params:,}")

    # ── Step 4: Training ──────────────────────────────────────────────────
    if not args.eval_only:
        logger.info("\n🏋️  STEP 4: Training...")
        from model.trainer import train
        history = train(model, train_loader, val_loader, device, CONFIG["training"])

        # Load best checkpoint for evaluation
        logger.info("Loading best checkpoint for evaluation...")
        model = load_model(model_path, tokenizer_vocab_size=len(tokenizer), device=device)

    # ── Step 5: Evaluation ────────────────────────────────────────────────
    logger.info("\n📊 STEP 5: Evaluating on test set...")
    metrics = evaluate_model(model, test_loader, device, save_plots=True)

    # ── Step 6: Save tokenizer ────────────────────────────────────────────
    if not args.eval_only:
        tokenizer_path = ROOT / CONFIG["paths"]["tokenizer"]
        tokenizer_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(tokenizer_path))
        logger.info(f"✅ Tokenizer saved to {tokenizer_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("🎉 PHISHGUARD TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"  F1 Score:   {metrics['f1']:.4f}")
    logger.info(f"  Recall:     {metrics['recall']:.4f}")
    logger.info(f"  Precision:  {metrics['precision']:.4f}")
    logger.info(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
    logger.info(f"  Targets met: {'✅ YES' if metrics.get('targets_passed') else '❌ NO — retrain with more data/epochs'}")
    logger.info("=" * 60)
    logger.info(f"\nNext step: Deploy the API with `uvicorn api.main:app --host 0.0.0.0 --port 8000`")


if __name__ == "__main__":
    main()
