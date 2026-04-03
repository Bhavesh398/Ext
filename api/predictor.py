"""
api/predictor.py
Loads trained model and runs inference for the PhishGuard API.
"""

import pathlib
import yaml
import logging
import time
import torch
from typing import Dict, Any, Optional
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from model.distilbert_classifier import PhishingClassifier, load_model
from model.dataset import get_tokenizer
from features.structural import extract_structural_features
from features.text_preprocessor import format_for_bert
from explainability.reasons import build_final_result

ROOT = pathlib.Path(__file__).parent.parent
CONFIG = yaml.safe_load(open(ROOT / "config.yaml"))
logger = logging.getLogger(__name__)


class PhishingPredictor:
    """
    Singleton predictor that loads the model once and serves predictions.
    Combines DistilBERT inference + structural feature analysis.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[PhishingClassifier] = None
        self.tokenizer = None
        self.max_length = CONFIG["model"]["max_length"]
        self._loaded = False

    def load(self):
        """Load model and tokenizer from artifacts directory."""
        if self._loaded:
            return

        model_path = ROOT / CONFIG["paths"]["model"]
        tokenizer_path = ROOT / CONFIG["paths"]["tokenizer"]

        logger.info(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = get_tokenizer()

        if tokenizer_path.exists():
            from transformers import DistilBertTokenizerFast
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(tokenizer_path))

        logger.info(f"Loading model from {model_path} on {self.device}")

        if not (model_path / "pytorch_model.bin").exists():
            logger.warning("No trained model found! Using untrained model for structure test.")
            self.model = PhishingClassifier(vocab_size=len(self.tokenizer))
        else:
            self.model = load_model(model_path, tokenizer_vocab_size=len(self.tokenizer), device=self.device)

        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info("✅ Model loaded successfully")

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize input text for DistilBERT."""
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].to(self.device),
        }

    def predict(self, email: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run full phishing analysis on an email.

        Args:
            email: Dict with keys: subject, from, replyTo, body, urls

        Returns:
            Complete analysis result matching Chrome extension API contract
        """
        if not self._loaded:
            self.load()

        t0 = time.time()

        # ── Step 1: Structural analysis ───────────────────────────────────
        try:
            structural_result = extract_structural_features(email)
        except Exception as e:
            logger.warning(f"Structural analysis failed: {e}")
            structural_result = {
                "flags": {}, "flag_list": [], "reasons": [],
                "senderAnalysis": {}, "urlAnalysis": {"total": 0, "suspicious": 0, "urls": []},
            }

        # ── Step 2: DistilBERT inference ──────────────────────────────────
        try:
            text = format_for_bert(
                subject=email.get("subject", ""),
                sender=email.get("from", ""),
                body=email.get("body", ""),
            )
            tokens = self._tokenize(text)

            with torch.no_grad():
                _, phishing_prob = self.model.predict(
                    tokens["input_ids"],
                    tokens["attention_mask"],
                )
            bert_prob = float(phishing_prob[0].cpu())

        except Exception as e:
            logger.warning(f"DistilBERT inference failed: {e}")
            # Fall back to structural-only scoring
            bert_prob = 0.5

        # ── Step 3: Combine and build result ─────────────────────────────
        result = build_final_result(bert_prob, structural_result, email)

        inference_ms = round((time.time() - t0) * 1000, 1)
        result["inferenceMs"] = inference_ms

        if inference_ms > CONFIG["inference"]["max_inference_time_ms"]:
            logger.warning(f"Slow inference: {inference_ms}ms (target: <200ms)")

        # ── Step 4: Log prediction ────────────────────────────────────────
        self._log_prediction(email, result)

        return result

    def _log_prediction(self, email: Dict[str, Any], result: Dict[str, Any]):
        """Log predictions for future retraining."""
        import json
        log_dir = ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        log_entry = {
            "subject": email.get("subject", "")[:100],
            "from": email.get("from", "")[:100],
            "score": result["score"],
            "label": result["label"],
            "flags": result["flags"],
            "inferenceMs": result.get("inferenceMs"),
        }
        with open(log_dir / "predictions.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")


# Global singleton instance
_predictor: Optional[PhishingPredictor] = None


def get_predictor() -> PhishingPredictor:
    """Get or create the global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = PhishingPredictor()
        _predictor.load()
    return _predictor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    predictor = PhishingPredictor()
    predictor.load()

    test_emails = [
        {
            "subject": "URGENT: Your PayPal account has been suspended",
            "from": "PayPal Support <noreply@paypa1.tk>",
            "replyTo": "harvest@evil.com",
            "body": "Dear customer, verify your account immediately or it will be permanently closed. Click here now.",
            "urls": ["http://paypa1.tk/verify"]
        },
        {
            "subject": "Team meeting tomorrow at 10am",
            "from": "John Smith <john@company.com>",
            "replyTo": "",
            "body": "Hi team, reminder that we have our weekly standup tomorrow. Please come prepared with updates.",
            "urls": []
        }
    ]

    for email in test_emails:
        result = predictor.predict(email)
        print(f"\nSubject: {email['subject'][:50]}")
        print(f"Score: {result['score']}/100 | Label: {result['label'].upper()} | Time: {result['inferenceMs']}ms")
        print(f"Reasons: {result['reasons'][0] if result['reasons'] else 'None'}")
