"""
features/text_preprocessor.py
Text cleaning and formatting for DistilBERT input.
"""

import re
import base64
import pathlib
import yaml
from typing import Optional
from bs4 import BeautifulSoup

ROOT = pathlib.Path(__file__).parent.parent
CONFIG = yaml.safe_load(open(ROOT / "config.yaml"))
MAX_BODY_CHARS = 800  # Keep body short to fit 512 tokens with subject/from


def decode_base64_parts(text: str) -> str:
    """Decode any base64-encoded segments found in text."""
    pattern = re.compile(r'[A-Za-z0-9+/]{40,}={0,2}')
    def try_decode(match):
        try:
            decoded = base64.b64decode(match.group()).decode("utf-8", errors="ignore")
            if decoded.isprintable():
                return decoded
        except Exception:
            pass
        return match.group()
    return pattern.sub(try_decode, text)


def strip_html(text: str) -> str:
    """Remove HTML tags and extract clean text."""
    try:
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "meta", "link", "head"]):
            tag.decompose()
        return soup.get_text(separator=" ")
    except Exception:
        return re.sub(r'<[^>]+>', ' ', text)


def remove_urls(text: str) -> str:
    """Replace URLs with [URL] token."""
    return re.sub(r'https?://\S+', '[URL]', text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace into single space."""
    text = re.sub(r'[\r\n\t\f\v]+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def remove_special_chars(text: str) -> str:
    """Remove non-printable and special characters."""
    text = re.sub(r'[^\x20-\x7E]', '', text)
    text = re.sub(r'[*#@!$%^&=~`|\\]{3,}', '', text)  # Remove repetitive symbols
    return text


def clean_email_text(raw_text: str) -> str:
    """
    Full cleaning pipeline for raw email text.

    Steps:
    1. Decode base64 parts
    2. Strip HTML
    3. Normalize whitespace
    4. Remove special characters
    5. Lowercase

    Args:
        raw_text: Raw email body text

    Returns:
        Cleaned plain text string
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return ""

    text = decode_base64_parts(raw_text)
    text = strip_html(text)
    text = normalize_whitespace(text)
    text = remove_special_chars(text)
    text = text.lower()
    return text.strip()


def format_for_bert(
    subject: Optional[str] = "",
    sender: Optional[str] = "",
    body: Optional[str] = "",
    max_body_chars: int = MAX_BODY_CHARS,
) -> str:
    """
    Format email components into DistilBERT input string.

    Format: "[SUBJECT] {subject} [FROM] {sender} [BODY] {body}"

    Special tokens [SUBJECT], [FROM], [BODY] help the model
    understand which part of the email it's reading.

    Args:
        subject: Email subject line
        sender: From header
        body: Email body text
        max_body_chars: Max characters for body (to fit 512 tokens)

    Returns:
        Formatted string ready for tokenization
    """
    subject_clean = clean_email_text(subject or "")[:200]
    sender_clean = clean_email_text(sender or "")[:100]
    body_clean = clean_email_text(body or "")[:max_body_chars]

    return f"[SUBJECT] {subject_clean} [FROM] {sender_clean} [BODY] {body_clean}"


def format_from_combined_text(text: str) -> str:
    """
    Format a combined email text string (when subject/from/body aren't separate).
    Used for Kaggle/CSV datasets that have a single text column.

    Args:
        text: Combined email text

    Returns:
        Cleaned and truncated text
    """
    cleaned = clean_email_text(text)
    return cleaned[:1200]  # ~512 tokens worth of text


if __name__ == "__main__":
    sample = {
        "subject": "URGENT: Your PayPal Account Has Been <b>SUSPENDED</b>!!!",
        "from": "PayPal Support <noreply@paypa1.tk>",
        "body": """<html><body>
            <p>Dear Customer,</p>
            <p>We detected <strong>unusual activity</strong> on your account.</p>
            <p>Click <a href='http://paypa1.tk/verify'>HERE</a> to verify immediately.</p>
        </body></html>"""
    }

    result = format_for_bert(sample["subject"], sample["from"], sample["body"])
    print("Formatted input:")
    print(result)
    print(f"\nLength: {len(result)} chars")
