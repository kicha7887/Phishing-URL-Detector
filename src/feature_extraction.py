"""
feature_extraction.py
---------------------
Extracts URL-based, domain-based, and security features from a given URL.
All features are numeric and suitable for ML model input.
"""

import re
import math
import urllib.parse
from collections import Counter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "update", "secure", "bank", "paypal",
    "signin", "confirm", "account", "password", "credential",
    "webscr", "free", "lucky", "service", "bonus", "ebayisapi",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entropy(s: str) -> float:
    """Shannon entropy of a string."""
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _has_ip_address(url: str) -> int:
    """Return 1 if URL contains an IPv4 address instead of a domain name."""
    pattern = r"((\d{1,3}\.){3}\d{1,3})"
    return 1 if re.search(pattern, url) else 0


def _count_subdomains(hostname: str) -> int:
    """Count the number of subdomains (dots minus 1 for TLD)."""
    parts = hostname.split(".")
    # e.g. www.paypal.com → 3 parts → 1 subdomain
    return max(0, len(parts) - 2)


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_features(url: str) -> dict:
    """
    Extract all engineered features from a URL string.

    Parameters
    ----------
    url : str
        The raw URL to analyse.

    Returns
    -------
    dict
        Feature name → numeric value.
    """
    features = {}

    # --- Normalise -----------------------------------------------------------
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        parsed = urllib.parse.urlparse("http://unknown")

    full_url   = url
    hostname   = parsed.hostname or ""
    path       = parsed.path or ""
    query      = parsed.query or ""

    # --- URL-level features --------------------------------------------------
    features["url_length"]           = len(full_url)
    features["num_dots"]             = full_url.count(".")
    features["num_hyphens"]          = full_url.count("-")
    features["num_underscores"]      = full_url.count("_")
    features["num_slashes"]          = full_url.count("/")
    features["num_question_marks"]   = full_url.count("?")
    features["num_equal_signs"]      = full_url.count("=")
    features["num_digits"]           = sum(c.isdigit() for c in full_url)
    features["has_ip_address"]       = _has_ip_address(full_url)
    features["has_at_symbol"]        = 1 if "@" in full_url else 0
    features["has_https"]            = 1 if full_url.startswith("https") else 0
    features["has_http"]             = 1 if full_url.startswith("http://") else 0
    features["url_entropy"]          = round(_entropy(full_url), 4)

    # --- Domain features -----------------------------------------------------
    features["hostname_length"]      = len(hostname)
    features["num_subdomains"]       = _count_subdomains(hostname)

    # --- Suspicious-keyword features -----------------------------------------
    lower_url = full_url.lower()
    features["suspicious_keyword_count"] = sum(
        lower_url.count(kw) for kw in SUSPICIOUS_KEYWORDS
    )
    features["has_suspicious_keyword"] = (
        1 if features["suspicious_keyword_count"] > 0 else 0
    )

    # --- Path / query features -----------------------------------------------
    features["path_length"]          = len(path)
    features["query_length"]         = len(query)
    features["num_params"]           = len(query.split("&")) if query else 0

    # --- Heuristic redirect proxy (no live request) --------------------------
    # Double-slash in path after the first segment can indicate redirect
    features["has_redirect"]         = 1 if "//" in path else 0

    return features


def get_feature_names() -> list:
    """Return ordered list of feature names (matches extract_features output)."""
    dummy = extract_features("http://example.com")
    return list(dummy.keys())
