"""
predict.py
----------
Loads the best saved model and makes predictions on new URLs.
Returns prediction label, confidence score, risk level, and feature values.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from feature_extraction import extract_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parent.parent
MODELS_DIR  = ROOT / "models"
MODEL_PATH  = MODELS_DIR / "best_phishing_model.pkl"
META_PATH   = MODELS_DIR / "model_meta.json"

# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------

def classify_risk(confidence_phishing: float) -> str:
    """Map phishing probability → risk label."""
    pct = confidence_phishing * 100
    if pct <= 30:
        return "Low Risk"
    elif pct <= 70:
        return "Medium Risk"
    return "High Risk"


RISK_COLOR = {
    "Low Risk":    "#00ff88",
    "Medium Risk": "#ffa500",
    "High Risk":   "#ff4444",
}

# ---------------------------------------------------------------------------
# Explainability helper
# ---------------------------------------------------------------------------

_EXPLANATIONS = {
    "url_length":               ("URL Length Extremely High",        lambda v: v > 75),
    "num_dots":                 ("Excessive Dots in URL",             lambda v: v > 4),
    "num_hyphens":              ("Multiple Hyphens Detected",         lambda v: v > 2),
    "num_underscores":          ("Underscores Present in URL",        lambda v: v > 0),
    "num_slashes":              ("Unusual Number of Slashes",         lambda v: v > 5),
    "num_question_marks":       ("Multiple Query Parameters",         lambda v: v > 1),
    "has_ip_address":           ("IP Address Used Instead of Domain", lambda v: v == 1),
    "has_at_symbol":            ("@ Symbol Found in URL",             lambda v: v == 1),
    "has_https":                ("No HTTPS (Insecure Connection)",    lambda v: v == 0),
    "num_subdomains":           ("Excessive Subdomains",              lambda v: v > 2),
    "suspicious_keyword_count": ("Contains Suspicious Keywords",      lambda v: v > 0),
    "has_suspicious_keyword":   ("Suspicious Keyword Detected",       lambda v: v == 1),
    "url_entropy":              ("High URL Entropy (Obfuscated URL)", lambda v: v > 4.5),
    "num_digits":               ("Unusually High Digit Count",        lambda v: v > 10),
    "hostname_length":          ("Long Hostname",                     lambda v: v > 30),
}


def explain_prediction(features: dict) -> list[str]:
    """Return human-readable reasons for a phishing classification."""
    reasons = []
    for feat, (msg, rule) in _EXPLANATIONS.items():
        if feat in features and rule(features[feat]):
            reasons.append(msg)
    return reasons or ["Probability threshold exceeded by model"]


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class PhishingPredictor:
    """Loads the saved model and provides predict() + threat_score()."""

    def __init__(self):
        self._model  = None
        self._meta   = {}
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'. "
                "Run `python src/train.py` first to train the model."
            )
        with open(MODEL_PATH, "rb") as f:
            self._model = pickle.load(f)
        if META_PATH.exists():
            with open(META_PATH) as f:
                self._meta = json.load(f)
        self._loaded = True

    # ------------------------------------------------------------------
    def predict(self, url: str) -> dict:
        """
        Analyse *url* and return full prediction payload.

        Returns
        -------
        dict:
            url, prediction, confidence, risk_level, risk_color,
            reasons, features, threat_score
        """
        self._load()

        # Extract features
        feats = extract_features(url)

        # Build feature vector (align to training columns if meta available)
        training_cols = self._meta.get("feature_names")
        if training_cols:
            row = {col: feats.get(col, 0) for col in training_cols}
        else:
            row = feats

        X = pd.DataFrame([row])

        # Predict
        proba       = self._model.predict_proba(X)[0]   # [P(legit), P(phish)]
        label_idx   = int(self._model.predict(X)[0])
        # Map model label to human label
        # In UCI dataset: 1 = phishing, -1 or 0 = legitimate
        classes = self._meta.get("label_encoder_classes", ["0", "1"])

        if len(proba) == 2:
            p_phishing = float(proba[1])
        else:
            p_phishing = float(proba[label_idx])

        is_phishing   = p_phishing >= 0.5
        prediction    = "PHISHING WEBSITE" if is_phishing else "LEGITIMATE WEBSITE"
        confidence    = p_phishing if is_phishing else (1.0 - p_phishing)
        risk_level    = classify_risk(p_phishing)
        threat_score  = self.threat_score(url, p_phishing, feats)
        reasons       = explain_prediction(feats) if is_phishing else []

        return {
            "url":          url,
            "prediction":   prediction,
            "is_phishing":  is_phishing,
            "p_phishing":   round(p_phishing, 4),
            "confidence":   round(confidence, 4),
            "risk_level":   risk_level,
            "risk_color":   RISK_COLOR[risk_level],
            "reasons":      reasons,
            "features":     feats,
            "threat_score": threat_score,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def threat_score(url: str, p_phishing: float, feats: dict) -> int:
        """
        Compute a 0-100 composite threat score.
        Combines model probability with URL heuristics.
        """
        score = p_phishing * 60  # base from model (up to 60)

        # Heuristic boosts (up to 40)
        if feats.get("has_ip_address"):   score += 10
        if feats.get("has_at_symbol"):    score += 8
        if not feats.get("has_https"):    score += 7
        if feats.get("num_subdomains", 0) > 2: score += 5
        if feats.get("suspicious_keyword_count", 0) > 0: score += 5
        if feats.get("url_length", 0) > 100: score += 5

        return min(100, int(score))


# ---------------------------------------------------------------------------
# Convenience singleton
# ---------------------------------------------------------------------------

predictor = PhishingPredictor()


def predict_url(url: str) -> dict:
    """Module-level shortcut: predict a single URL."""
    return predictor.predict(url)
