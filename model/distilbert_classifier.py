"""
model/distilbert_classifier.py
DistilBERT-based phishing email classifier architecture.
"""

import pathlib
import yaml
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertPreTrainedModel
from typing import Optional, Tuple

ROOT = pathlib.Path(__file__).parent.parent
CONFIG = yaml.safe_load(open(ROOT / "config.yaml"))


class PhishingClassifier(nn.Module):
    """
    DistilBERT-based binary classifier for phishing detection.

    Architecture:
        DistilBERT → [CLS] token → Dropout → Linear(768→256) →
        ReLU → Dropout → Linear(256→2) → logits

    The [CLS] token's 768-dimensional embedding represents the
    entire email and is used for classification.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        num_labels: int = 2,
        dropout: float = 0.3,
        hidden_size: int = 256,
        vocab_size: Optional[int] = None,
    ):
        super().__init__()
        model_name = model_name or CONFIG["model"]["base_model"]
        self.num_labels = num_labels

        # DistilBERT backbone
        self.distilbert = DistilBertModel.from_pretrained(model_name)

        # Resize embeddings if custom tokens were added
        if vocab_size:
            self.distilbert.resize_token_embeddings(vocab_size)

        # Classification head
        self.pre_classifier = nn.Linear(768, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(0.1)
        self.relu = nn.ReLU()

        # Initialize classification head weights
        self._init_weights()

    def _init_weights(self):
        """Initialize linear layer weights."""
        nn.init.xavier_uniform_(self.pre_classifier.weight)
        nn.init.zeros_(self.pre_classifier.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Ground truth labels [batch_size] (optional)

        Returns:
            (loss, logits) — loss is None if labels not provided
        """
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # [CLS] token representation
        hidden_state = outputs.last_hidden_state[:, 0]  # [batch_size, 768]

        # Classification head
        x = self.dropout1(hidden_state)
        x = self.pre_classifier(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.classifier(x)  # [batch_size, 2]

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return loss, logits

    def get_probabilities(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get softmax probabilities."""
        with torch.no_grad():
            _, logits = self.forward(input_ids, attention_mask)
            return torch.softmax(logits, dim=-1)

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions and phishing probabilities.

        Returns:
            (predicted_class, phishing_probability)
        """
        probs = self.get_probabilities(input_ids, attention_mask)
        phishing_probs = probs[:, 1]  # Probability of class 1 (phishing)
        predicted = torch.argmax(probs, dim=-1)
        return predicted, phishing_probs


def load_model(
    model_path: pathlib.Path,
    tokenizer_vocab_size: Optional[int] = None,
    device: Optional[str] = None,
) -> PhishingClassifier:
    """
    Load a saved PhishingClassifier from disk.

    Args:
        model_path: Path to saved model directory
        tokenizer_vocab_size: Vocab size after adding custom tokens
        device: 'cuda', 'cpu', or None (auto-detect)

    Returns:
        Loaded PhishingClassifier in eval mode
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PhishingClassifier(vocab_size=tokenizer_vocab_size)
    state_dict = torch.load(model_path / "pytorch_model.bin", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def save_model(model: PhishingClassifier, save_path: pathlib.Path):
    """Save model weights to disk."""
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / "pytorch_model.bin")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = PhishingClassifier()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 2
    seq_len = 512
    input_ids = torch.randint(0, 30522, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)

    with torch.no_grad():
        loss, logits = model(input_ids, attention_mask)

    print(f"Output logits shape: {logits.shape}")
    print(f"Sample probabilities: {torch.softmax(logits, dim=-1)}")
