"""Locked final evaluation on the held-out test split.

Everything a reviewer should need to judge the model, in one report:
headline metrics with bootstrap confidence intervals, per-class breakdown,
global and per-class calibration, the coverage/accuracy curve the routing
layer depends on, and an explicit comparison against the majority baseline.

Confidence intervals matter here because the differences being discussed
(0.7294 vs 0.7313, or a model against a 0.5946 baseline) are meaningless
without knowing the sampling error on a 3.5k-row split.

Usage:
    python final_eval.py --predictions outputs/modernbert_test.csv --name modernbert_clean_v1_test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

BOOTSTRAP_N = 2000
RNG_SEED = 42


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric, n: int = BOOTSTRAP_N):
    """Percentile bootstrap 95% CI over ticket resampling."""
    rng = np.random.default_rng(RNG_SEED)
    size = len(y_true)
    samples = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, size, size)
        samples[i] = metric(y_true[idx], y_pred[idx])
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def ece_and_bins(df: pd.DataFrame, n_bins: int = 10):
    edges = [i / n_bins for i in range(n_bins + 1)]
    bins = pd.cut(df["confidence"].clip(0, 1), bins=edges, include_lowest=True)
    rows, score = [], 0.0
    for interval, group in df.groupby(bins, observed=True):
        if group.empty:
            continue
        conf, acc = float(group["confidence"].mean()), float(group["correct"].mean())
        score += len(group) / len(df) * abs(acc - conf)
        rows.append({"bin": str(interval), "n": len(group),
                     "mean_confidence": round(conf, 4), "accuracy": round(acc, 4),
                     "gap": round(acc - conf, 4)})
    return round(score, 4), pd.DataFrame(rows)


def per_class_ece(df: pd.DataFrame) -> pd.DataFrame:
    """Global ECE can hide a queue that is badly calibrated on its own."""
    rows = []
    for label, group in df.groupby("predicted_label"):
        score, _ = ece_and_bins(group)
        rows.append({"predicted_label": label, "n": len(group),
                     "mean_confidence": round(float(group["confidence"].mean()), 4),
                     "precision": round(float(group["correct"].mean()), 4),
                     "ece": score})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def coverage_curve(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in [round(0.05 * i, 2) for i in range(20)]:
        auto = df[df["confidence"] >= threshold]
        if auto.empty:
            continue
        rows.append({
            "threshold": threshold,
            "coverage": round(len(auto) / len(df), 4),
            "accuracy_on_auto": round(accuracy_score(auto["true_label"], auto["predicted_label"]), 4),
            "macro_f1_on_auto": round(f1_score(auto["true_label"], auto["predicted_label"],
                                               average="macro", zero_division=0), 4),
            "n_auto": len(auto),
            "n_errors_auto": int((auto["correct"] == 0).sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked final evaluation.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--name", default="final")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    df["correct"] = (df["true_label"] == df["predicted_label"]).astype(int)
    out_dir = args.output_dir or args.predictions.parent

    y_true = df["true_label"].to_numpy()
    y_pred = df["predicted_label"].to_numpy()

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    acc_ci = bootstrap_ci(y_true, y_pred, accuracy_score)
    macro_ci = bootstrap_ci(y_true, y_pred,
                            lambda a, b: f1_score(a, b, average="macro", zero_division=0))
    weighted_ci = bootstrap_ci(y_true, y_pred,
                               lambda a, b: f1_score(a, b, average="weighted", zero_division=0))

    majority_label = df["true_label"].value_counts().idxmax()
    majority = float((df["true_label"] == majority_label).mean())
    ece, reliability = ece_and_bins(df)

    metrics = {
        "name": args.name,
        "n": int(len(df)),
        "accuracy": round(float(acc), 4),
        "accuracy_ci95": [round(acc_ci[0], 4), round(acc_ci[1], 4)],
        "macro_f1": round(float(macro), 4),
        "macro_f1_ci95": [round(macro_ci[0], 4), round(macro_ci[1], 4)],
        "weighted_f1": round(float(weighted), 4),
        "weighted_f1_ci95": [round(weighted_ci[0], 4), round(weighted_ci[1], 4)],
        "ece": ece,
        "majority_class": majority_label,
        "majority_baseline": round(majority, 4),
        "lift_over_majority": round(float(acc) - majority, 4),
        "beats_majority_at_95ci": bool(acc_ci[0] > majority),
        "bootstrap_resamples": BOOTSTRAP_N,
    }
    if "latency_ms" in df.columns:
        latency = pd.to_numeric(df["latency_ms"], errors="coerce").dropna()
        if not latency.empty:
            metrics["latency_p50_ms"] = round(float(latency.median()), 2)
            metrics["latency_p95_ms"] = round(float(latency.quantile(0.95)), 2)

    report = pd.DataFrame(classification_report(
        y_true, y_pred, output_dict=True, zero_division=0)).transpose()
    coverage = coverage_curve(df)
    class_ece = per_class_ece(df)

    (out_dir / f"{args.name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    report.to_csv(out_dir / f"{args.name}_classification_report.csv")
    reliability.to_csv(out_dir / f"{args.name}_reliability.csv", index=False)
    coverage.to_csv(out_dir / f"{args.name}_coverage_curve.csv", index=False)
    class_ece.to_csv(out_dir / f"{args.name}_per_class_calibration.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print("\nPer-class:")
    print(report.round(4).to_string())
    print("\nReliability:")
    print(reliability.to_string(index=False))
    print("\nPer-class calibration (on predictions made):")
    print(class_ece.to_string(index=False))
    print("\nCoverage curve (selected):")
    print(coverage[coverage.threshold.isin([0.5, 0.65, 0.7, 0.8, 0.9])].to_string(index=False))
    print(f"\nWrote 5 files to {out_dir}")


if __name__ == "__main__":
    main()
