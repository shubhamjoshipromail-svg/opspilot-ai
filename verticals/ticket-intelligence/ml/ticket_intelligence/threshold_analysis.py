"""Analyze confidence thresholds for human-in-the-loop ticket routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from data_utils import OUTPUT_DIR


def threshold_values() -> list[float]:
    return [round(value * 0.05, 2) for value in range(20)]


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {path}")
    df = pd.read_csv(path)
    required = {"true_label", "predicted_label", "confidence"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Predictions CSV missing required columns: {missing}")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    if "top1_top2_margin" not in df.columns and {"top_1_probability", "top_2_probability"}.issubset(df.columns):
        df["top1_top2_margin"] = (
            pd.to_numeric(df["top_1_probability"], errors="coerce").fillna(0.0)
            - pd.to_numeric(df["top_2_probability"], errors="coerce").fillna(0.0)
        ).round(4)
    return df


def metric_row(df: pd.DataFrame, mask: pd.Series, threshold: float, margin_threshold: float | None = None) -> dict:
    routed = df[mask]
    if len(routed) == 0:
        accuracy = None
        macro_f1 = None
    else:
        accuracy = round(float(accuracy_score(routed["true_label"], routed["predicted_label"])), 4)
        macro_f1 = round(float(f1_score(routed["true_label"], routed["predicted_label"], average="macro", zero_division=0)), 4)
    row = {
        "threshold": threshold,
        "coverage": round(float(mask.mean()), 4),
        "accuracy_on_auto_routed": accuracy,
        "macro_f1_on_auto_routed": macro_f1,
        "number_auto_routed": int(mask.sum()),
        "number_sent_to_review": int((~mask).sum()),
    }
    if margin_threshold is not None:
        row["margin_threshold"] = margin_threshold
    return row


def analyze_thresholds(df: pd.DataFrame, margin_threshold: float | None = None) -> pd.DataFrame:
    rows = []
    for threshold in threshold_values():
        mask = df["confidence"] >= threshold
        if margin_threshold is not None:
            if "top1_top2_margin" not in df.columns:
                raise ValueError("Margin threshold requested but top1_top2_margin is not available.")
            mask = mask & (pd.to_numeric(df["top1_top2_margin"], errors="coerce").fillna(0.0) >= margin_threshold)
        rows.append(metric_row(df, mask, threshold, margin_threshold))
    return pd.DataFrame(rows)


def write_plot(results: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(results["threshold"], results["coverage"], marker="o", label="Coverage")
    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("Coverage / auto-route rate")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(results["threshold"], results["accuracy_on_auto_routed"], marker="s", color="tab:green", label="Accuracy")
    ax2.plot(results["threshold"], results["macro_f1_on_auto_routed"], marker="^", color="tab:orange", label="Macro-F1")
    ax2.set_ylabel("Quality on auto-routed subset")
    ax2.set_ylim(0, 1.05)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    ax1.set_title("Human Review Threshold Tradeoff")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def default_output_prefix(predictions_path: Path) -> str:
    name = predictions_path.stem
    if name.endswith("_predictions"):
        return name[: -len("_predictions")]
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze confidence thresholds for auto-routing vs human review.")
    parser.add_argument("--predictions", type=Path, required=True, help="Predictions CSV from train_modernbert.py.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--margin-threshold", type=float, default=None, help="Optional top1-top2 margin threshold.")
    args = parser.parse_args()

    df = load_predictions(args.predictions)
    results = analyze_thresholds(df, args.margin_threshold)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or default_output_prefix(args.predictions)
    suffix = "_coverage_by_threshold"
    if args.margin_threshold is not None:
        suffix = f"_coverage_by_threshold_margin_{str(args.margin_threshold).replace('.', '_')}"

    csv_path = output_dir / f"{prefix}{suffix}.csv"
    png_path = output_dir / f"{prefix}{suffix}.png"
    json_path = output_dir / f"{prefix}{suffix}.json"
    results.to_csv(csv_path, index=False)
    write_plot(results, png_path)
    json_path.write_text(json.dumps(results.to_dict(orient="records"), indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "png": str(png_path), "json": str(json_path), "rows": len(results)}, indent=2))


if __name__ == "__main__":
    main()
