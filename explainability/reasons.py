"""
explainability/reasons.py
Combines DistilBERT score + structural features into human-readable output.
"""

import pathlib
import yaml
from typing import Dict, List, Any, Tuple

ROOT = pathlib.Path(__file__).parent.parent
CONFIG = yaml.safe_load(open(ROOT / "config.yaml"))

THRESHOLDS = CONFIG["thresholds"]


def bert_score_to_label(score: int) -> str:
    """Convert numeric score to label string."""
    if score >= THRESHOLDS["phishing"]:
        return "phishing"
    elif score >= THRESHOLDS["suspicious"]:
        return "suspicious"
    return "safe"


def probability_to_score(prob: float) -> int:
    """Convert DistilBERT phishing probability (0-1) to score (0-100)."""
    return min(100, max(0, round(prob * 100)))


def generate_reasons(
    structural_flags: Dict[str, bool],
    bert_score: int,
    structural_reasons: List[str],
) -> List[str]:
    """
    Generate final list of human-readable reasons.

    Combines:
    1. Structural flag reasons (rule-based, always explicit)
    2. DistilBERT-based reasons (when structural flags don't explain the score)

    Args:
        structural_flags: Dict of flag_name → bool
        bert_score: Phishing score from DistilBERT (0-100)
        structural_reasons: Pre-built reasons from structural.py

    Returns:
        List of up to 4 reasons, sorted by impact
    """
    reasons = list(structural_reasons)  # Start with structural reasons

    # Add model-level reason if score is high but few structural flags
    flag_count = sum(1 for v in structural_flags.values() if v)

    if bert_score >= THRESHOLDS["phishing"] and flag_count == 0:
        reasons.append("Language patterns and writing style closely match known phishing email templates")

    elif bert_score >= THRESHOLDS["suspicious"] and flag_count <= 1:
        reasons.append("Writing style shows statistical characteristics of suspicious emails")

    elif bert_score >= THRESHOLDS["phishing"] and flag_count > 0:
        reasons.append(f"Multiple risk signals detected — combined phishing score: {bert_score}/100")

    # If nothing found and email is safe
    if not reasons:
        reasons.append("No significant phishing indicators detected")

    # Return top 4 reasons (most impactful first)
    return reasons[:4]


def combine_scores(
    bert_prob: float,
    structural_score: int,
    bert_weight: float = 0.65,
) -> int:
    """
    Combine DistilBERT probability with structural feature score.

    Weighted blend:
    - DistilBERT: 65% weight (learned patterns)
    - Structural: 35% weight (rule-based signals)

    Args:
        bert_prob: DistilBERT phishing probability (0.0–1.0)
        structural_score: Structural feature score (0–60, from score_from_flags)
        bert_weight: Weight for DistilBERT score (0.0–1.0)

    Returns:
        Combined phishing score (0–100)
    """
    bert_score = bert_prob * 100
    struct_weight = 1.0 - bert_weight

    # Scale structural score from 0-60 to 0-100
    struct_normalized = min(100, (structural_score / 60) * 100)

    combined = (bert_score * bert_weight) + (struct_normalized * struct_weight)
    return min(100, max(0, round(combined)))


def build_final_result(
    bert_prob: float,
    structural_result: Dict[str, Any],
    email_meta: Dict[str, str],
) -> Dict[str, Any]:
    """
    Build the complete final result dict returned by the API.

    Args:
        bert_prob: DistilBERT phishing probability (0.0–1.0)
        structural_result: Output from extract_structural_features()
        email_meta: Dict with subject, from, replyTo, urls

    Returns:
        Complete result dict matching the Chrome extension API contract
    """
    import sys
    sys.path.append(str(ROOT))
    from features.structural import score_from_flags

    structural_flags = structural_result.get("flags", {})
    structural_reasons = structural_result.get("reasons", [])
    structural_score = score_from_flags(structural_flags)

    # Combine scores
    final_score = combine_scores(bert_prob, structural_score)
    label = bert_score_to_label(final_score)

    # Generate reasons
    reasons = generate_reasons(structural_flags, final_score, structural_reasons)

    return {
        "score": final_score,
        "label": label,
        "reasons": reasons,
        "flags": structural_result.get("flag_list", []),
        "senderAnalysis": structural_result.get("senderAnalysis", {}),
        "urlAnalysis": structural_result.get("urlAnalysis", {}),
        "confidence": round(float(bert_prob), 4),
        "bertScore": probability_to_score(bert_prob),
        "structuralScore": structural_score,
    }


if __name__ == "__main__":
    # Test
    structural_result = {
        "flags": {"DOMAIN_MISMATCH": True, "URGENT_LANGUAGE": True},
        "flag_list": ["DOMAIN_MISMATCH", "URGENT_LANGUAGE"],
        "reasons": [
            "Sender display name impersonates PayPal but domain doesn't match",
            'Contains urgency language: "verify", "immediately"',
        ],
        "senderAnalysis": {
            "displayName": "PayPal Support",
            "domain": "paypa1.tk",
            "replyTo": None,
            "domainMismatch": True,
        },
        "urlAnalysis": {"total": 2, "suspicious": 1, "urls": []},
    }

    result = build_final_result(
        bert_prob=0.91,
        structural_result=structural_result,
        email_meta={},
    )

    print(f"Score: {result['score']}/100")
    print(f"Label: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("Reasons:")
    for r in result["reasons"]:
        print(f"  • {r}")
