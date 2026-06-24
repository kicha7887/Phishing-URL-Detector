"""
evaluate.py
-----------
Load saved model(s) and produce evaluation metrics + visualisations.
Run as a standalone script or import individual functions.

Usage
-----
    python src/evaluate.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)

# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parent.parent
MODELS_DIR  = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
DATA_PATH   = ROOT / "data" / "processed" / "phishing_data.csv"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_model(name: str):
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_test_data():
    """Load and prepare test split (same seed as train.py)."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(DATA_PATH)
    candidates = ["Result", "label", "class", "phishing", "status", "target"]
    label_col = next((c for c in candidates if c in df.columns), df.columns[-1])

    le = LabelEncoder()
    df[label_col] = le.fit_transform(df[label_col].astype(str))
    drop_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != label_col]
    df.drop(columns=drop_cols, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    X = df.drop(columns=[label_col])
    y = df[label_col]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_test, y_test


def evaluate_all():
    """Evaluate all available saved models and save plots."""
    if not DATA_PATH.exists():
        print("⚠  No dataset found. Run train.py first.")
        return

    X_test, y_test = _load_test_data()

    model_names = ["random_forest", "xgboost", "best_phishing_model"]
    seen = set()

    fig_cm, axes_cm = plt.subplots(1, 2, figsize=(12, 5))
    ax_idx = 0

    for name in model_names:
        model = _load_model(name)
        if model is None or id(model) in seen:
            continue
        seen.add(id(model))

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        print(f"\n{'='*55}")
        print(f"  {name}")
        print(f"{'='*55}")
        print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
        print(f"  ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

        # Confusion matrix
        if ax_idx < 2:
            ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred,
                display_labels=["Legitimate", "Phishing"],
                ax=axes_cm[ax_idx],
                colorbar=False,
            )
            axes_cm[ax_idx].set_title(name)
            ax_idx += 1

    plt.tight_layout()
    cm_path = REPORTS_DIR / "confusion_matrices.png"
    fig_cm.savefig(cm_path, dpi=150)
    plt.close(fig_cm)
    print(f"\n📊 Confusion matrices saved → {cm_path}")


if __name__ == "__main__":
    evaluate_all()
