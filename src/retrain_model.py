"""
retrain_model.py
----------------
Automated retraining pipeline. Run this script whenever new labelled data is
added to data/processed/phishing_data.csv. It retrains, evaluates, and
replaces the saved model files.

Usage
-----
    python src/retrain_model.py
    python src/retrain_model.py --data /path/to/new_data.csv --append
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------

def backup_models():
    """Archive current models before overwriting."""
    models_dir = ROOT / "models"
    backup_dir = ROOT / "models" / "backups" / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)

    for pkl in models_dir.glob("*.pkl"):
        shutil.copy(pkl, backup_dir / pkl.name)
    print(f"🗄  Models backed up → {backup_dir}")


def append_data(new_csv: Path):
    """Append new_csv rows to the main processed dataset."""
    import pandas as pd

    main_csv = ROOT / "data" / "processed" / "phishing_data.csv"
    if not main_csv.exists():
        shutil.copy(new_csv, main_csv)
        print(f"📂 New dataset created at {main_csv}")
        return

    existing = pd.read_csv(main_csv)
    new_data  = pd.read_csv(new_csv)
    combined  = pd.concat([existing, new_data], ignore_index=True)
    combined.drop_duplicates(inplace=True)
    combined.to_csv(main_csv, index=False)
    print(f"➕ Appended {len(new_data)} rows  →  total: {len(combined)} rows")


def retrain(data_path: Path | None = None, append: bool = False, new_csv: Path | None = None):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from train import train

    print("\n" + "="*55)
    print("  AI Phishing Detector – Retraining Pipeline")
    print(f"  Started: {datetime.utcnow().isoformat(timespec='seconds')} UTC")
    print("="*55)

    # Append new data if provided
    if append and new_csv:
        append_data(new_csv)

    # Backup existing models
    backup_models()

    # Retrain
    train(data_path)

    print("\n✅ Retraining complete. New best model saved.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain the phishing detection model.")
    parser.add_argument("--data",   type=Path, default=None, help="Path to dataset CSV.")
    parser.add_argument("--new",    type=Path, default=None, help="New CSV to append before retraining.")
    parser.add_argument("--append", action="store_true",     help="Append --new CSV to existing dataset.")
    args = parser.parse_args()
    retrain(data_path=args.data, append=args.append, new_csv=args.new)
