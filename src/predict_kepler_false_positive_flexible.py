"""
Flexible predictor for the Kepler false-positive model.

This version lets your input CSV contain fewer columns than the model was trained with.
Any missing feature columns are automatically created as blank values, then the model's
imputer fills them with the median values learned during training.

Important:
    The prediction will still run, but it becomes less reliable when many important
    features are missing. Try to provide at least koi_period, koi_depth, koi_prad,
    and koi_model_snr when possible.

Example:
    python predict_kepler_false_positive_flexible.py \
        --model kepler_fp_model_results/kepler_fp_parameter_only.joblib \
        --input minimal_planets_input.csv \
        --output predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained .joblib model file")
    parser.add_argument("--input", required=True, help="CSV file containing new planet/KOI parameters")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=0.50, help="Probability threshold for false-positive prediction")
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.input)

    try:
        feature_names = list(model.feature_names_in_)
    except AttributeError as exc:
        raise RuntimeError(
            "This model does not store feature_names_in_. Make sure it was trained with a pandas DataFrame."
        ) from exc

    missing = [c for c in feature_names if c not in df.columns]
    extra = [c for c in df.columns if c not in feature_names]

    # Add missing columns as NaN. The trained model pipeline has a median imputer,
    # so missing values will be filled using training-set medians.
    for col in missing:
        df[col] = np.nan

    # Keep the exact feature order used during training.
    X = df[feature_names].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("This model does not support predict_proba().")

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= args.threshold).astype(int)

    result = pd.read_csv(args.input)
    result["false_positive_probability"] = probabilities
    result["predicted_false_positive"] = predictions
    result["prediction_label"] = pd.Series(predictions).map(
        {1: "LIKELY FALSE POSITIVE", 0: "LIKELY NOT FALSE POSITIVE"}
    )
    result["missing_features_filled_by_model"] = len(missing)
    result["features_used_by_model"] = len(feature_names)

    output_path = Path(args.output)
    result.to_csv(output_path, index=False)

    print(f"Saved predictions to {output_path.resolve()}")
    print("\nPrediction result:")
    print(result[["false_positive_probability", "prediction_label", "missing_features_filled_by_model", "features_used_by_model"]].to_string(index=False))

    if missing:
        print("\nWarning: your input file was missing these features, so the model filled them with training-set median values:")
        print(", ".join(missing))
    if extra:
        print("\nNote: your input file had extra columns not used by this model:")
        print(", ".join(extra))


if __name__ == "__main__":
    main()
