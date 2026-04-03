"""
model/dataset.py
PyTorch Dataset class for phishing email classification.
"""

import pathlib
import torch
import pandas as pd
import yaml
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast
from typing import Optional, Dict
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from features.text_preprocessor import format_for_bert, format_from_combined_text

ROOT = pathlib.Path(__file__).parent.parent
CONFIG = yaml.safe_load(open(ROOT / "config.yaml"))


class PhishingDataset(Dataset):
    """
    PyTorch Dataset for phishing email classification.

    Tokenizes email text using DistilBERT tokenizer and returns
    input_ids, attention_mask, and label tensors.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: DistilBertTokenizerFast,
        max_length: int = 512,
        has_separate_cols: bool = False,
    ):
        """
        Args:
            df: DataFrame with columns [text, label] or [subject, from, body, label]
            tokenizer: HuggingFace DistilBERT tokenizer
            max_length: Max token length (512 for DistilBERT)
            has_separate_cols: If True, use subject/from/body cols separately
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_separate_cols = has_separate_cols
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)

        # Pre-format all texts
        self.texts = self._prepare_texts(df)

    def _prepare_texts(self, df: pd.DataFrame):
        """Format all texts for DistilBERT input."""
        texts = []
        for _, row in df.iterrows():
            if self.has_separate_cols and all(c in df.columns for c in ["subject", "from", "body"]):
                text = format_for_bert(
                    subject=row.get("subject", ""),
                    sender=row.get("from", ""),
                    body=row.get("body", ""),
                )
            else:
                text = format_from_combined_text(str(row.get("text", "")))
            texts.append(text)
        return texts

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": label,
        }


def get_tokenizer(model_name: Optional[str] = None) -> DistilBertTokenizerFast:
    """Load DistilBERT tokenizer, adding custom tokens."""
    model_name = model_name or CONFIG["model"]["base_model"]
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    # Add special email tokens so model understands email structure
    special_tokens = ["[SUBJECT]", "[FROM]", "[BODY]", "[URL]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    return tokenizer


def get_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: DistilBertTokenizerFast,
    batch_size: Optional[int] = None,
) -> tuple:
    """
    Create DataLoader objects for train, val, and test splits.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    batch_size = batch_size or CONFIG["training"]["batch_size"]
    max_length = CONFIG["model"]["max_length"]

    train_dataset = PhishingDataset(train_df, tokenizer, max_length)
    val_dataset = PhishingDataset(val_df, tokenizer, max_length)
    test_dataset = PhishingDataset(test_df, tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick test
    tokenizer = get_tokenizer()
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    sample_df = pd.DataFrame({
        "text": [
            "URGENT verify your PayPal account now or it will be suspended",
            "Hi team, meeting tomorrow at 10am please come prepared",
        ],
        "label": [1, 0]
    })

    dataset = PhishingDataset(sample_df, tokenizer)
    sample = dataset[0]
    print(f"input_ids shape: {sample['input_ids'].shape}")
    print(f"attention_mask shape: {sample['attention_mask'].shape}")
    print(f"label: {sample['label']}")
