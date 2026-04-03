"""
features/structural.py
Rule-based structural feature extraction from emails.
These features power the "reasons" shown to users in PhishGuard.
"""

import re
import pathlib
import yaml
from typing import Dict, List, Any
from urllib.parse import urlparse

ROOT = pathlib.Path(__file__).parent.parent
CONFIG = yaml.safe_load(open(ROOT / "config.yaml"))

URGENT_KEYWORDS: List[str] = CONFIG["urgent_keywords"]
SUSPICIOUS_TLDS: List[str] = CONFIG["suspicious_tlds"]
KNOWN_BRANDS: List[str] = CONFIG["known_brands"]

URL_REGEX = re.compile(r'https?://[^\s"\'<>)\]]+', re.IGNORECASE)
IP_REGEX = re.compile(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
EMAIL_REGEX = re.compile(r'[\w.+-]+@([\w.-]+\.[a-z]{2,})', re.IGNORECASE)
CAPS_RATIO_THRESHOLD = 0.45


# ─── Domain Helpers ───────────────────────────────────────────────────────────

def extract_domain(email_str: str) -> str:
    """Extract sending domain from From header string."""
    match = EMAIL_REGEX.search(email_str or "")
    return match.group(1).lower() if match else ""


def extract_display_name(email_str: str) -> str:
    """Extract display name from From header."""
    match = re.match(r'^([^<@\n]+?)(?:\s*<|$)', (email_str or "").strip())
    return match.group(1).strip().strip('"') if match else ""


def get_root_domain(domain: str) -> str:
    """
    Extract root domain from a full domain string.
    Handles multi-part TLDs like .co.in, .com.au etc.

    Examples:
        esisc.nse.co.in  -> nse.co.in
        mail.paypal.com  -> paypal.com
        nse.co.in        -> nse.co.in
    """
    # Known multi-part TLDs (extend as needed)
    MULTI_TLDS = {".co.in", ".com.au", ".co.uk", ".org.uk", ".net.in", ".gov.in"}

    domain = domain.lower().strip()
    for tld in MULTI_TLDS:
        if domain.endswith(tld):
            # Get one label before the multi-TLD
            prefix = domain[: -len(tld)]
            label = prefix.split(".")[-1]
            return f"{label}{tld}"

    # Standard: take last two parts
    parts = domain.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return domain


def is_subdomain_of(child: str, parent: str) -> bool:
    """
    Return True if child is a subdomain of parent or equal to parent.

    Examples:
        esisc.nse.co.in, nse.co.in  -> True
        mail.paypal.com, paypal.com -> True
        evil.tk, paypal.com         -> False
    """
    child = child.lower().strip(".")
    parent = parent.lower().strip(".")
    return child == parent or child.endswith(f".{parent}")


# ─── Reply-To Mismatch (Layered) ──────────────────────────────────────────────

def check_replyto_mismatch(sender_domain: str, replyto_domain: str) -> tuple[bool, str]:
    """
    Layered reply-to mismatch detection.
    Returns (is_mismatch: bool, reason: str)

    Layer 1: Identical domains → safe
    Layer 2: Subdomain of same root + clean TLD → safe (NSE case)
    Layer 3: Suspicious TLD on reply-to → always flag
    Layer 4: Brand impersonation (different root) → always flag
    Layer 5: Different root domains → flag
    """
    if not sender_domain or not replyto_domain:
        return False, ""

    sender_root = get_root_domain(sender_domain)
    replyto_root = get_root_domain(replyto_domain)

    # Layer 1: Identical → safe
    if sender_domain == replyto_domain:
        return False, ""

    # Layer 3: Suspicious TLD on reply-to → always flag regardless of root
    if any(replyto_domain.endswith(tld) for tld in SUSPICIOUS_TLDS):
        return True, (
            f"Reply-To address ({replyto_domain}) uses a high-risk domain extension"
        )

    # Layer 2: Reply-to is a subdomain (or equal root) of sender → safe
    # e.g. esisc.nse.co.in vs nse.co.in
    if is_subdomain_of(replyto_domain, sender_root) or is_subdomain_of(sender_domain, replyto_root):
        return False, ""

    # Layer 4: Same root domain but different subdomains with clean TLD → safe
    if sender_root == replyto_root:
        return False, ""

    # Layer 5: Completely different root domains → flag
    return True, (
        f"Reply-To address ({replyto_domain}) differs from sender domain ({sender_domain})"
    )


# ─── URL Helpers ─────────────────────────────────────────────────────────────

def extract_urls(text: str) -> List[str]:
    """Extract all URLs from text."""
    return list(set(URL_REGEX.findall(text or "")))


def get_url_domain(url: str) -> str:
    """Safely extract domain from URL."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def has_suspicious_tld(domain: str) -> bool:
    """Check if domain ends with a suspicious TLD."""
    return any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS)


def has_ip_url(urls: List[str]) -> bool:
    """Check if any URL uses an IP address instead of domain."""
    return any(IP_REGEX.match(u) for u in urls)


# ─── Text Helpers ────────────────────────────────────────────────────────────

def brand_in_text(text: str) -> bool:
    """Check if text mentions any known brand."""
    text_lower = text.lower()
    return any(brand in text_lower for brand in KNOWN_BRANDS)


def find_urgent_keywords(text: str) -> List[str]:
    """Find urgent/phishing keywords in text."""
    text_lower = text.lower()
    return [kw for kw in URGENT_KEYWORDS if kw in text_lower]


def caps_ratio(text: str) -> float:
    """Calculate ratio of uppercase letters."""
    if not text:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def has_html_form(text: str) -> bool:
    """Check if email body contains HTML form elements."""
    return bool(re.search(r'<form[\s>]', text or "", re.IGNORECASE))


def count_external_images(text: str) -> int:
    """Count external image tags."""
    return len(re.findall(r'<img[^>]+src=["\']https?://', text or "", re.IGNORECASE))


# ─── Main Feature Extractor ───────────────────────────────────────────────────

def extract_structural_features(email: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract all structural features from an email dict.

    Args:
        email: dict with keys: subject, from, replyTo, body, urls

    Returns:
        dict with flags (bool), values, and reasons (list of strings)
    """
    subject      = str(email.get("subject", "") or "")
    from_header  = str(email.get("from", "") or "")
    reply_to     = str(email.get("replyTo", "") or "")
    body         = str(email.get("body", "") or "")
    urls         = email.get("urls", []) or []

    if not urls:
        urls = extract_urls(body)

    full_text = f"{subject} {body}"

    # ── Extract components ────────────────────────────────────────────────
    sender_domain   = extract_domain(from_header)
    display_name    = extract_display_name(from_header)
    replyto_domain  = extract_domain(reply_to) if reply_to else ""

    url_domains     = [get_url_domain(u) for u in urls]
    suspicious_urls = [u for u, d in zip(urls, url_domains) if has_suspicious_tld(d)]
    ip_urls         = [u for u in urls if IP_REGEX.match(u)]

    urgent_found    = find_urgent_keywords(full_text)
    subject_caps    = caps_ratio(subject)
    brand_in_display = brand_in_text(display_name)
    brand_in_domain  = brand_in_text(sender_domain)
    domain_mismatch  = brand_in_display and not brand_in_domain

    # Layered reply-to check (replaces naive != comparison)
    replyto_is_mismatch, replyto_reason = check_replyto_mismatch(
        sender_domain, replyto_domain
    )

    has_form   = has_html_form(body)
    ext_images = count_external_images(body)

    # ── Build flags & reasons ─────────────────────────────────────────────
    flags   = {}
    reasons = []

    if domain_mismatch:
        flags["DOMAIN_MISMATCH"] = True
        reasons.append(
            f"Sender display name impersonates a known brand "
            f"but domain '{sender_domain}' doesn't match"
        )

    if has_suspicious_tld(sender_domain):
        flags["SUSPICIOUS_TLD"] = True
        reasons.append(f"Sender uses high-risk domain extension: {sender_domain}")

    if replyto_is_mismatch:
        flags["REPLY_TO_MISMATCH"] = True
        reasons.append(replyto_reason)

    if urgent_found:
        flags["URGENT_LANGUAGE"] = True
        sample = '", "'.join(urgent_found[:3])
        reasons.append(f'Contains urgency language: "{sample}"')

    if suspicious_urls:
        flags["SUSPICIOUS_URLS"] = True
        reasons.append(
            f"Contains {len(suspicious_urls)} link(s) with suspicious domain extensions"
        )

    if ip_urls:
        flags["IP_BASED_URL"] = True
        reasons.append("Contains links using raw IP addresses instead of domain names")

    if len(urls) > 7:
        flags["MANY_URLS"] = True
        reasons.append(
            f"Unusually high number of links ({len(urls)}) for a legitimate email"
        )

    if subject_caps > CAPS_RATIO_THRESHOLD and len(subject) > 5:
        flags["EXCESSIVE_CAPS"] = True
        reasons.append("Subject line uses excessive capital letters — common phishing tactic")

    if has_form:
        flags["HTML_FORM"] = True
        reasons.append("Email contains HTML form elements — possible credential harvesting")

    if ext_images > 3:
        flags["TRACKING_IMAGES"] = True
        reasons.append(f"Contains {ext_images} external images — possible tracking pixels")

    return {
        "flags": flags,
        "flag_list": list(flags.keys()),
        "reasons": reasons,
        "senderAnalysis": {
            "displayName":   display_name,
            "domain":        sender_domain,
            "replyTo":       reply_to or None,
            "domainMismatch": domain_mismatch,
        },
        "urlAnalysis": {
            "total":      len(urls),
            "suspicious": len(suspicious_urls),
            "hasIpUrls":  bool(ip_urls),
            "urls":       urls[:5],
        },
        "meta": {
            "urgentKeywordsFound": urgent_found,
            "subjectCapsRatio":    round(subject_caps, 3),
            "hasHtmlForm":         has_form,
            "externalImages":      ext_images,
        },
    }


# ─── Structural Score ────────────────────────────────────────────────────────

def score_from_flags(flags: Dict[str, bool]) -> int:
    """
    Calculate a structural score contribution (0-60) based on flags.
    Combined with DistilBERT score for final scoring.
    """
    weights = {
        "DOMAIN_MISMATCH":   30,
        "SUSPICIOUS_TLD":    20,
        "REPLY_TO_MISMATCH": 15,
        "URGENT_LANGUAGE":   15,
        "SUSPICIOUS_URLS":   20,
        "IP_BASED_URL":      25,
        "MANY_URLS":          8,
        "EXCESSIVE_CAPS":     8,
        "HTML_FORM":         20,
        "TRACKING_IMAGES":    5,
    }
    total = sum(weights.get(flag, 0) for flag in flags)
    return min(total, 60)


# ─── Quick Test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        {
            "name": "NSE legitimate email (should NOT flag reply-to)",
            "email": {
                "subject": "NSE Circular: Action required immediately",
                "from": "NSE India <noreply@nse.co.in>",
                "replyTo": "support@esisc.nse.co.in",
                "body": "Please review the attached circular.",
                "urls": ["https://nse.co.in/circular1", "https://nse.co.in/circular2"],
            },
        },
        {
            "name": "Phishing with suspicious reply-to TLD (SHOULD flag)",
            "email": {
                "subject": "URGENT: Verify your PayPal account immediately",
                "from": "PayPal Support <noreply@paypal.com>",
                "replyTo": "harvest@paypal.com.evil.tk",
                "body": "Click here to verify your account or it will be suspended.",
                "urls": ["http://paypa1.tk/verify"],
            },
        },
        {
            "name": "Phishing lookalike subdomain (SHOULD flag)",
            "email": {
                "subject": "Your account has been suspended",
                "from": "PayPal <support@paypa1.tk>",
                "replyTo": "harvest@mail.paypa1.tk",
                "body": "Verify your account immediately to avoid suspension.",
                "urls": ["http://paypa1.tk/login"],
            },
        },
        {
            "name": "Legit bank email with many links (should NOT flag reply-to)",
            "email": {
                "subject": "Your HDFC Bank statement is ready",
                "from": "HDFC Bank <alerts@hdfcbank.com>",
                "replyTo": "noreply@mail.hdfcbank.com",
                "body": "Your monthly statement is ready. View it online.",
                "urls": [f"https://hdfcbank.com/link{i}" for i in range(10)],
            },
        },
    ]

    for test in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {test['name']}")
        result = extract_structural_features(test["email"])
        score  = score_from_flags(result["flags"])
        print(f"Flags:  {result['flag_list']}")
        print(f"Score contribution: {score}")
        print("Reasons:")
        for r in result["reasons"]:
            print(f"  • {r}")