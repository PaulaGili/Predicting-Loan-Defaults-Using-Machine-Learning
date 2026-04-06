"""
predict.py — run inference on new loans using a saved model bundle

Usage:
    python predict.py --model models/LightGBM_best.joblib --input new_loans.csv
    python predict.py --model models/LightGBM_best.joblib --input new_loans.csv --output preds.csv
    python predict.py --model models/LightGBM_best.joblib --input new_loans.csv --threshold 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.validation import ValidationError, validate_input


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict loan defaults on new data")
    p.add_argument("--model",     required=True, help="path to .joblib bundle")
    p.add_argument("--input",     required=True, help="CSV with new loan applications")
    p.add_argument("--output",    default="predictions.csv")
    p.add_argument("--threshold", type=float, default=None,
                   help="override decision threshold (default: use the one saved in the bundle)")
    return p.parse_args()


def load_bundle(model_path: str) -> dict:
    path = Path(model_path)
    if not path.exists():
        print(f"Error: model not found at {path}", file=sys.stderr)
        sys.exit(1)
    bundle = joblib.load(path)
    missing = {"model", "feature_pipeline", "feature_names", "optimal_threshold"} - set(bundle.keys())
    if missing:
        print(f"Error: bundle missing keys: {missing}", file=sys.stderr)
        sys.exit(1)
    return bundle


def run_predictions(bundle: dict, df_raw: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
    t = threshold if threshold is not None else bundle["optimal_threshold"]
    X = bundle["feature_pipeline"].transform(df_raw).values.astype(float)
    proba = bundle["model"].predict_proba(X)[:, 1]
    return pd.DataFrame({
        "default_probability": proba.round(4),
        "predicted_default":   (proba >= t).astype(int),
    }, index=df_raw.index)


def main() -> None:
    args = parse_args()

    print(f"Loading model from {args.model}...")
    bundle = load_bundle(args.model)
    t = args.threshold if args.threshold is not None else bundle["optimal_threshold"]
    print(f"  threshold = {t:.3f}")

    print(f"Reading {args.input}...")
    df_raw = pd.read_csv(args.input, low_memory=False)
    print(f"  {len(df_raw):,} rows")

    try:
        validate_input(df_raw)
    except ValidationError as e:
        print(f"\nValidation failed:\n{e}", file=sys.stderr)
        sys.exit(1)

    results = run_predictions(bundle, df_raw, args.threshold)
    n = results["predicted_default"].sum()
    print(f"  {n:,} predicted defaults ({n / len(results):.1%})")

    results.to_csv(args.output, index=True)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
