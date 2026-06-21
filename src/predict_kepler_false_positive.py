"""
Use a trained Kepler false-positive model to predict whether new KOI-like inputs
are likely false positives.

Before using this script, train the model first with:
    python train_kepler_false_positive_model.py

Then predict with:
    python predict_kepler_false_positive.py --model kepler_fp_model_results/kepler_fp_parameter_only.joblib --input new_planets.csv

The input CSV must contain the same feature columns used by the model.
For the parameter-only model, columns can include:
    koi_period, koi_impact, koi_duration, koi_depth, koi_ror, koi_srho,
    koi_prad, koi_sma, koi_incl, koi_teq, koi_insol, koi_dor,
    koi_max_sngle_ev, koi_max_mult_ev, koi_model_snr, koi_count,
    koi_num_transits, koi_steff, koi_slogg, koi_smet, koi_srad,
    koi_smass, koi_kepmag

For the parameters-plus-flags model, also include:
    koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
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

    # The model remembers which feature names it was trained with.
    try:
        feature_names = list(model.feature_names_in_)
    except AttributeError:
        raise RuntimeError(
            "This model does not store feature_names_in_. Make sure it was trained with a pandas DataFrame."
        )

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(
            "Your input CSV is missing these required columns:\n" + "\n".join(missing)
        )

    X = df[feature_names].copy()

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        raise RuntimeError("This model does not support predict_proba().")

    predictions = (probabilities >= args.threshold).astype(int)

    result = df.copy()
    result["false_positive_probability"] = probabilities
    result["predicted_false_positive"] = predictions
    result["prediction_label"] = result["predicted_false_positive"].map(
        {1: "LIKELY FALSE POSITIVE", 0: "LIKELY NOT FALSE POSITIVE"}
    )

    output_path = Path(args.output)
    result.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path.resolve()}")
    print(result[["false_positive_probability", "prediction_label"]].to_string(index=False))


if __name__ == "__main__":
    main()
