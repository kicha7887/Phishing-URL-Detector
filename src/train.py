"""
train.py
--------
Trains Random Forest and XGBoost classifiers on a phishing URL dataset,
selects the best model by ROC-AUC, and saves it to models/.

Usage
-----
    python src/train.py --data data/processed/phishing_data.csv
    python src/train.py          # uses default path above
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import warnings
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# Optional XGBoost import (graceful fallback)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠  XGBoost not installed – only Random Forest will be trained.")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dataset(csv_path: Path) -> tuple[pd.DataFrame, str]:
    """
    Load dataset and return (DataFrame, label_column_name).
    Tries to infer the label column automatically.
    """
    print(f"📂 Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Shape: {df.shape}")

    # Common label column names in UCI / Kaggle phishing datasets
    candidates = ["Result", "label", "class", "phishing", "status", "target"]
    label_col = None
    for col in candidates:
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        # Fall back to last column
        label_col = df.columns[-1]
    print(f"   Label column: '{label_col}'")
    return df, label_col


def _prepare(df: pd.DataFrame, label_col: str):
    """Clean, encode, split."""
    df = df.copy()

    # Drop non-numeric columns except the label
    drop_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != label_col]
    if drop_cols:
        print(f"   Dropping non-numeric columns: {drop_cols}")
        df.drop(columns=drop_cols, inplace=True)

    # Encode label to 0 / 1
    le = LabelEncoder()
    df[label_col] = le.fit_transform(df[label_col].astype(str))

    # Fill NaN
    df.fillna(df.median(numeric_only=True), inplace=True)

    X = df.drop(columns=[label_col])
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, list(X.columns), le


def _evaluate(name: str, model, X_test, y_test) -> dict:
    """Compute and print evaluation metrics."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model":     name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print(f"\n{'='*55}")
    print(f"  {name} Results")
    print(f"{'='*55}")
    for k, v in metrics.items():
        if k not in ("model", "confusion_matrix"):
            print(f"  {k:<15}: {v}")
    print(f"  Confusion Matrix:\n  {np.array(metrics['confusion_matrix'])}")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

    return metrics


def _save_model(model, name: str) -> Path:
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Saved {name} → {path}")
    return path


# ---------------------------------------------------------------------------
# Main training logic
# ---------------------------------------------------------------------------

def train(csv_path: Path | None = None):
    if csv_path is None:
        csv_path = DATA_DIR / "phishing_data.csv"

    if not csv_path.exists():
        print(
            f"\n⚠  Dataset not found at '{csv_path}'.\n"
            "   Please download a phishing URL dataset and place it at:\n"
            f"   {csv_path}\n"
            "   Recommended sources:\n"
            "     • https://archive.ics.uci.edu/ml/datasets/phishing+websites\n"
            "     • https://www.kaggle.com/datasets/eswarchandt/phishing-website-detector\n"
        )
        return

    df, label_col = _load_dataset(csv_path)
    X_train, X_test, y_train, y_test, feature_names, le = _prepare(df, label_col)

    results   = []
    all_models = {}

    # --- Random Forest -------------------------------------------------------
    print("\n🌲 Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_metrics = _evaluate("Random Forest", rf, X_test, y_test)
    results.append(rf_metrics)
    all_models["random_forest"] = rf
    _save_model(rf, "random_forest")

    # --- XGBoost -------------------------------------------------------------
    if XGBOOST_AVAILABLE:
        print("\n🚀 Training XGBoost …")
        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
        xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        xgb_metrics = _evaluate("XGBoost", xgb, X_test, y_test)
        results.append(xgb_metrics)
        all_models["xgboost"] = xgb
        _save_model(xgb, "xgboost")

    # --- Select best model by ROC-AUC ----------------------------------------
    best = max(results, key=lambda m: m["roc_auc"])
    best_name = "random_forest" if best["model"] == "Random Forest" else "xgboost"
    best_model = all_models[best_name]

    _save_model(best_model, "best_phishing_model")
    print(f"\n🏆 Best model: {best['model']}  (ROC-AUC = {best['roc_auc']})")

    # --- Save feature names for inference ------------------------------------
    meta = {
        "feature_names": feature_names,
        "label_encoder_classes": list(le.classes_),
        "best_model": best["model"],
    }
    with open(MODELS_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # --- Save metrics report -------------------------------------------------
    report = {
        "models":     results,
        "best_model": best,
    }
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"📊 Metrics saved → {REPORTS_DIR / 'metrics.json'}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train phishing detection models.")
    parser.add_argument(
        "--data", type=Path, default=None,
        help="Path to processed CSV dataset (default: data/processed/phishing_data.csv)"
    )
    args = parser.parse_args()
    train(args.data)

    