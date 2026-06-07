"""
Train a Kepler false-positive prediction model and generate visual scatterplot diagnostics.

Target:
    is_false_positive = 1 if koi_disposition == 'FALSE POSITIVE'
    is_false_positive = 0 if koi_disposition == 'CONFIRMED' or 'CANDIDATE'

Two models are trained:
    1. parameter_only: planet/transit/stellar parameters only.
    2. parameters_plus_flags: same parameters plus NASA archive FP diagnostic flags.

Main outputs:
    kepler_fp_model_results/metrics_summary.csv
    kepler_fp_model_results/confusion_matrix_parameter_only.csv
    kepler_fp_model_results/confusion_matrix_parameters_plus_flags.csv
    kepler_fp_model_results/feature_importance_parameter_only.csv
    kepler_fp_model_results/feature_importance_parameters_plus_flags.csv
    kepler_fp_model_results/kepler_fp_parameter_only.joblib
    kepler_fp_model_results/kepler_fp_parameters_plus_flags.joblib

New figure outputs:
    kepler_fp_model_results/figures/<model>_depth_vs_radius_probability.png
    kepler_fp_model_results/figures/<model>_snr_vs_probability.png
    kepler_fp_model_results/figures/<model>_noise_proxy_vs_confidence.png
    kepler_fp_model_results/figures/<model>_confidence_vs_accuracy.png
    kepler_fp_model_results/figures/<model>_feature_importance_scatter.png

Install:
    pip install pandas numpy scikit-learn joblib matplotlib

Run:
    python train_kepler_false_positive_model_with_plots.py

Optional local CSV:
    python train_kepler_false_positive_model_with_plots.py --csv cumulative.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import quote_plus

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

TARGET_COL = "koi_disposition"
RANDOM_STATE = 42

PARAMETER_FEATURES = [
    "koi_period", "koi_impact", "koi_duration", "koi_depth", "koi_ror",
    "koi_srho", "koi_prad", "koi_sma", "koi_incl", "koi_teq",
    "koi_insol", "koi_dor", "koi_max_sngle_ev", "koi_max_mult_ev",
    "koi_model_snr", "koi_count", "koi_num_transits", "koi_steff",
    "koi_slogg", "koi_smet", "koi_srad", "koi_smass", "koi_kepmag"
]

FLAG_FEATURES = ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]


def nasa_url(columns):
    query = "select " + ",".join(columns) + " from cumulative"
    return "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=" + quote_plus(query) + "&format=csv"


def load_data(csv_path=None):
    needed = [TARGET_COL] + PARAMETER_FEATURES + FLAG_FEATURES
    if csv_path:
        return pd.read_csv(csv_path, comment="#")
    url = nasa_url(needed)
    print("Downloading NASA Exoplanet Archive selected columns...")
    return pd.read_csv(url, comment="#")


def prepare_data(df):
    df = df[df[TARGET_COL].isin(["FALSE POSITIVE", "CONFIRMED", "CANDIDATE"])].copy()
    df["is_false_positive"] = (df[TARGET_COL] == "FALSE POSITIVE").astype(int)
    for col in PARAMETER_FEATURES + FLAG_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def usable(df, cols):
    return [c for c in cols if c in df.columns and df[c].notna().any()]


def build_model(features):
    preprocess = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), features)],
        remainder="drop"
    )
    model = RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def safe_log10(series):
    """Safe log transform for positive-skew parameters such as depth, radius, SNR, period."""
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return np.log10(np.clip(values, a_min=0, a_max=None) + 1)


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_depth_vs_radius(test_df, name, metrics, figures_dir):
    """Scatterplot: transit depth vs planet radius, colored by model FP probability."""
    required = {"koi_depth", "koi_prad"}
    if not required.issubset(test_df.columns):
        return

    data = test_df.dropna(subset=["koi_depth", "koi_prad", "false_positive_probability"]).copy()
    if data.empty:
        return

    x = safe_log10(data["koi_depth"])
    y = safe_log10(data["koi_prad"])
    color = data["false_positive_probability"]

    if "koi_model_snr" in data.columns:
        snr = pd.to_numeric(data["koi_model_snr"], errors="coerce").fillna(0)
        size = 15 + 55 * (safe_log10(snr) / max(safe_log10(snr).max(), 1e-6))
    else:
        size = 28

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    scatter = ax.scatter(x, y, c=color, s=size, alpha=0.62, edgecolors="none")
    ax.axhline(np.log10(20 + 1), linestyle="--", linewidth=1.2, label="~20 Earth radii warning line")
    ax.set_xlabel("log10(Transit depth ppm + 1)")
    ax.set_ylabel("log10(Planet radius Earth radii + 1)")
    ax.set_title(
        f"{name}: Transit Depth vs Planet Radius\n"
        f"Color = predicted FP probability | Accuracy={metrics['accuracy']:.3f}, AUC={metrics['roc_auc']:.3f}"
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Predicted false-positive probability")
    ax.legend(loc="best")
    save_figure(fig, figures_dir / f"{name}_depth_vs_radius_probability.png")


def plot_snr_vs_probability(test_df, name, metrics, figures_dir):
    """Scatterplot: SNR against model predicted probability, with correct/wrong markers."""
    if "koi_model_snr" not in test_df.columns:
        return

    data = test_df.dropna(subset=["koi_model_snr", "false_positive_probability", "prediction_correct"]).copy()
    if data.empty:
        return

    x = safe_log10(data["koi_model_snr"])
    y = data["false_positive_probability"]
    correct = data["prediction_correct"].astype(bool)

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.scatter(x[correct], y[correct], s=26, alpha=0.55, label="Correct prediction")
    ax.scatter(x[~correct], y[~correct], s=42, alpha=0.85, marker="x", label="Wrong prediction")
    ax.axhline(0.5, linestyle="--", linewidth=1.2, label="Decision threshold = 0.5")
    ax.set_xlabel("log10(Model signal-to-noise ratio + 1)")
    ax.set_ylabel("Predicted false-positive probability")
    ax.set_title(
        f"{name}: SNR vs Predicted False-Positive Probability\n"
        f"Wrong points reveal where the model makes mistakes | F1={metrics['f1_false_positive']:.3f}"
    )
    ax.legend(loc="best")
    save_figure(fig, figures_dir / f"{name}_snr_vs_probability.png")


def plot_noise_proxy_vs_confidence(test_df, name, metrics, figures_dir):
    """Scatterplot: noise proxy from SNR vs prediction confidence.

    Since a direct noise column is not always available, this plot uses 1/(SNR+1) as a simple
    noise/uncertainty proxy. Higher values mean lower SNR and therefore noisier detections.
    """
    if "koi_model_snr" not in test_df.columns:
        return

    data = test_df.dropna(subset=["koi_model_snr", "model_confidence", "prediction_correct"]).copy()
    if data.empty:
        return

    snr = pd.to_numeric(data["koi_model_snr"], errors="coerce").clip(lower=0)
    noise_proxy = 1 / (snr + 1)
    correct = data["prediction_correct"].astype(bool)

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.scatter(noise_proxy[correct], data.loc[correct, "model_confidence"], s=25, alpha=0.55, label="Correct prediction")
    ax.scatter(noise_proxy[~correct], data.loc[~correct, "model_confidence"], s=45, alpha=0.85, marker="x", label="Wrong prediction")
    ax.set_xscale("log")
    ax.set_xlabel("Noise proxy = 1 / (SNR + 1), log scale")
    ax.set_ylabel("Model confidence = |probability - 0.5| × 2")
    ax.set_title(
        f"{name}: Noise Proxy vs Model Confidence\n"
        f"Accuracy={metrics['accuracy']:.3f}; low confidence near noisy data means prediction should be cautious"
    )
    ax.legend(loc="best")
    save_figure(fig, figures_dir / f"{name}_noise_proxy_vs_confidence.png")


def plot_confidence_vs_accuracy(test_df, name, metrics, figures_dir):
    """Scatterplot: binned model confidence vs realized accuracy.

    This shows whether the model is actually more correct when it is more confident.
    The point size is proportional to the number of test objects in that confidence bin.
    """
    data = test_df.dropna(subset=["model_confidence", "prediction_correct"]).copy()
    if data.empty:
        return

    bins = np.linspace(0, 1, 11)
    data["confidence_bin"] = pd.cut(data["model_confidence"], bins=bins, include_lowest=True)
    grouped = data.groupby("confidence_bin", observed=False).agg(
        mean_confidence=("model_confidence", "mean"),
        accuracy=("prediction_correct", "mean"),
        count=("prediction_correct", "size"),
    ).dropna()
    if grouped.empty:
        return

    sizes = 40 + 260 * grouped["count"] / grouped["count"].max()

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.scatter(grouped["mean_confidence"], grouped["accuracy"], s=sizes, alpha=0.72)
    for _, row in grouped.iterrows():
        ax.text(row["mean_confidence"], row["accuracy"] + 0.015, str(int(row["count"])), ha="center", fontsize=8)
    ax.plot([0, 1], [0.5, 1], linestyle="--", linewidth=1.1, label="Desired upward trend")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("Mean model confidence in bin")
    ax.set_ylabel("Actual accuracy in bin")
    ax.set_title(
        f"{name}: Model Confidence vs Real Accuracy\n"
        "Number labels show how many test KOIs are in each confidence bin"
    )
    ax.legend(loc="lower right")
    save_figure(fig, figures_dir / f"{name}_confidence_vs_accuracy.png")


def plot_feature_importance_scatter(importance_df, name, figures_dir):
    """Scatterplot: permutation importance mean vs uncertainty."""
    if importance_df.empty:
        return

    top = importance_df.sort_values("importance_auc_drop", ascending=False).head(15).copy()
    fig, ax = plt.subplots(figsize=(8.7, 6.4))
    ax.scatter(top["importance_auc_drop"], top["importance_std"], s=80, alpha=0.72)
    for _, row in top.iterrows():
        ax.text(row["importance_auc_drop"], row["importance_std"], "  " + row["feature"], va="center", fontsize=8)
    ax.axvline(0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Mean AUC drop when feature is randomly shuffled")
    ax.set_ylabel("Importance uncertainty / standard deviation")
    ax.set_title(f"{name}: Feature Importance vs Importance Uncertainty")
    save_figure(fig, figures_dir / f"{name}_feature_importance_scatter.png")


def make_test_prediction_frame(X_test, y_test, pred, prob):
    """Return one table combining test inputs, labels, predictions, and model confidence."""
    test_df = X_test.copy()
    test_df["actual_false_positive"] = y_test.to_numpy()
    test_df["predicted_false_positive"] = pred
    test_df["false_positive_probability"] = prob
    test_df["prediction_correct"] = (test_df["actual_false_positive"] == test_df["predicted_false_positive"])
    test_df["model_confidence"] = np.abs(test_df["false_positive_probability"] - 0.5) * 2
    return test_df


def evaluate(name, X, y, outdir, make_plots=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    model = build_model(list(X.columns))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "precision_false_positive": precision_score(y_test, pred, zero_division=0),
        "recall_false_positive": recall_score(y_test, pred, zero_division=0),
        "f1_false_positive": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
    }

    pd.DataFrame(confusion_matrix(y_test, pred),
                 index=["actual_not_fp", "actual_fp"],
                 columns=["predicted_not_fp", "predicted_fp"]).to_csv(outdir / f"confusion_matrix_{name}.csv")

    test_df = make_test_prediction_frame(X_test, y_test, pred, prob)
    test_df.to_csv(outdir / f"test_predictions_{name}.csv", index=False)

    perm = permutation_importance(model, X_test, y_test, n_repeats=10, scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=-1)
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance_auc_drop": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_auc_drop", ascending=False)
    importance_df.to_csv(outdir / f"feature_importance_{name}.csv", index=False)

    joblib.dump(model, outdir / f"kepler_fp_{name}.joblib")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_validate(model, X, y, cv=cv, scoring={"accuracy":"accuracy", "balanced_accuracy":"balanced_accuracy", "f1":"f1", "roc_auc":"roc_auc"}, n_jobs=-1)
    pd.DataFrame({
        "metric": ["accuracy", "balanced_accuracy", "f1", "roc_auc"],
        "mean": [np.mean(cv_scores["test_accuracy"]), np.mean(cv_scores["test_balanced_accuracy"]), np.mean(cv_scores["test_f1"]), np.mean(cv_scores["test_roc_auc"])],
        "std": [np.std(cv_scores["test_accuracy"]), np.std(cv_scores["test_balanced_accuracy"]), np.std(cv_scores["test_f1"]), np.std(cv_scores["test_roc_auc"])],
    }).to_csv(outdir / f"cross_validation_{name}.csv", index=False)

    if make_plots:
        figures_dir = outdir / "figures"
        plot_depth_vs_radius(test_df, name, metrics, figures_dir)
        plot_snr_vs_probability(test_df, name, metrics, figures_dir)
        plot_noise_proxy_vs_confidence(test_df, name, metrics, figures_dir)
        plot_confidence_vs_accuracy(test_df, name, metrics, figures_dir)
        plot_feature_importance_scatter(importance_df, name, figures_dir)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Optional local cumulative CSV file")
    parser.add_argument("--outdir", default="kepler_fp_model_results")
    parser.add_argument("--no-plots", action="store_true", help="Train model without generating figure PNG files")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = prepare_data(load_data(args.csv))
    y = df["is_false_positive"]
    param_cols = usable(df, PARAMETER_FEATURES)
    flag_cols = usable(df, FLAG_FEATURES)

    print("Rows used:", len(df))
    print("False-positive fraction:", round(float(y.mean()), 4))
    print("Parameter features used:", param_cols)
    print("Flag features used:", flag_cols)

    results = []
    results.append(evaluate("parameter_only", df[param_cols], y, outdir, make_plots=not args.no_plots))
    results.append(evaluate("parameters_plus_flags", df[param_cols + flag_cols], y, outdir, make_plots=not args.no_plots))
    pd.DataFrame(results).to_csv(outdir / "metrics_summary.csv", index=False)

    print(pd.DataFrame(results).to_string(index=False))
    print("Saved model outputs to", outdir.resolve())
    if not args.no_plots:
        print("Saved figures to", (outdir / "figures").resolve())


if __name__ == "__main__":
    main()
