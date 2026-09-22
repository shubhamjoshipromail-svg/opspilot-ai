"""Score a Jev predictions CSV against the published ModernBERT baselines.

Reports the three metrics already in metrics.json (accuracy / macro-F1 /
weighted-F1), calibration (ECE + reliability bins), and the coverage-vs-accuracy
curve that decides the human-in-the-loop question.

With --remap-clean-v1, a 10-class run is collapsed into the clean_v1 space so
it can be compared to the ~0.73 run without a second API pass.

Usage:
    python score_jev.py --predictions outputs/jev_parent_queue_predictions.csv
    python score_jev.py --predictions outputs/jev_parent_queue_predictions.csv --remap-clean-v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from taxonomy import to_clean_v1

# Published ModernBERT-base parent_queue results on this same fixed split.
# original_10 is from ML_RESULTS_REPORT.md section 23; clean_v1 is the ~0.73
# figure in the repo description and is approximate until the Colab metrics
# JSON is recovered.
BASELINES = {
    "original_10": {"accuracy": 0.5229, "macro_f1": 0.4725, "weighted_f1": 0.5158,
                    "source": "ML_RESULTS_REPORT.md s23"},
    "clean_v1": {"accuracy": 0.73, "macro_f1": None, "weighted_f1": None,
                 "source": "repo description (approximate)"},
    "tfidf_10": {"accuracy": 0.4409, "macro_f1": 0.4121, "weighted_f1": 0.4430,
                 "source": "model_leaderboard.csv"},
}

THRESHOLDS = [round(0.05 * i, 2) for i in range(20)]


def expected_calibration_error(df: pd.DataFrame, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    """ECE plus the per-bin table. Jev's pitch is that this number is small."""
    conf = df["confidence"].clip(0.0, 1.0)
    bins = pd.cut(conf, bins=[i / n_bins for i in range(n_bins + 1)],
                  include_lowest=True, right=True)
    rows = []
    ece = 0.0
    total = len(df)
    for interval, group in df.groupby(bins, observed=True):
        if group.empty:
            continue
        mean_conf = float(group["confidence"].mean())
        acc = float(group["correct"].mean())
        weight = len(group) / total
        ece += weight * abs(acc - mean_conf)
        rows.append({
            "bin": str(interval),
            "n": len(group),
            "mean_confidence": round(mean_conf, 4),
            "accuracy": round(acc, 4),
            "gap": round(acc - mean_conf, 4),
        })
    return round(ece, 4), pd.DataFrame(rows)


def coverage_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy on auto-routed tickets at each confidence cutoff."""
    rows = []
    for threshold in THRESHOLDS:
        routed = df[df["confidence"] >= threshold]
        if routed.empty:
            rows.append({"threshold": threshold, "auto_route_rate": 0.0,
                         "human_review_rate": 1.0, "accuracy_on_auto": None,
                         "macro_f1_on_auto": None, "n_auto": 0})
            continue
        rows.append({
            "threshold": threshold,
            "auto_route_rate": round(len(routed) / len(df), 4),
            "human_review_rate": round(1 - len(routed) / len(df), 4),
            "accuracy_on_auto": round(accuracy_score(routed["true_label"], routed["predicted_label"]), 4),
            "macro_f1_on_auto": round(f1_score(routed["true_label"], routed["predicted_label"],
                                               average="macro", zero_division=0), 4),
            "n_auto": len(routed),
        })
    return pd.DataFrame(rows)


def accuracy_at_budget(df: pd.DataFrame, budgets=(0.1, 0.2, 0.3, 0.5)) -> pd.DataFrame:
    """Fix the share of tickets a human sees; report accuracy on the rest.

    This is the operational comparison: at an equal review budget, which system
    routes the remainder more accurately?
    """
    ranked = df.sort_values("confidence", ascending=False)
    rows = []
    for budget in budgets:
        keep = int(round(len(ranked) * (1 - budget)))
        if keep <= 0:
            continue
        auto = ranked.head(keep)
        rows.append({
            "human_review_budget": budget,
            "auto_routed": keep,
            "accuracy_on_auto": round(accuracy_score(auto["true_label"], auto["predicted_label"]), 4),
            "macro_f1_on_auto": round(f1_score(auto["true_label"], auto["predicted_label"],
                                               average="macro", zero_division=0), 4),
            "min_confidence_kept": round(float(auto["confidence"].min()), 4),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Jev against the ModernBERT baselines.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--remap-clean-v1", action="store_true",
                        help="Collapse a 10-class run into the clean_v1 label space.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)

    if args.remap_clean_v1:
        df["true_label"] = df["true_label"].map(to_clean_v1)
        df["predicted_label"] = df["predicted_label"].map(to_clean_v1)

    # Infer the label space from the data rather than assuming: a run made with
    # --taxonomy clean_v1 is already merged and must not be scored against the
    # 10-class baseline.
    present = set(df["true_label"]) | set(df["predicted_label"])
    merged_only = {"technical_product_support", "customer_general"}
    taxonomy = "clean_v1" if present & merged_only else "original_10"

    df["correct"] = (df["true_label"] == df["predicted_label"]).astype(int)

    # Always-guess-the-largest-class, the floor any real system must clear.
    majority = df["true_label"].value_counts(normalize=True).iloc[0]

    out_dir = args.output_dir or args.predictions.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"jev_{taxonomy}"

    metrics = {
        "taxonomy": taxonomy,
        "n": len(df),
        "accuracy": round(accuracy_score(df["true_label"], df["predicted_label"]), 4),
        "macro_f1": round(f1_score(df["true_label"], df["predicted_label"],
                                   average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(df["true_label"], df["predicted_label"],
                                      average="weighted", zero_division=0), 4),
    }
    ece, reliability = expected_calibration_error(df)
    metrics["ece"] = ece
    if "latency_ms" in df.columns:
        latency = pd.to_numeric(df["latency_ms"], errors="coerce").dropna()
        if not latency.empty:
            metrics["latency_p50_ms"] = round(float(latency.median()), 1)
            metrics["latency_p95_ms"] = round(float(latency.quantile(0.95)), 1)
    if "input_tokens" in df.columns:
        tokens = pd.to_numeric(df["input_tokens"], errors="coerce").dropna()
        if not tokens.empty:
            total_tokens = float(tokens.sum())
            metrics["total_input_tokens"] = int(total_tokens)
            # $0.042 per million input tokens, output free.
            metrics["estimated_cost_usd"] = round(total_tokens / 1e6 * 0.042, 6)

    metrics["majority_class_baseline"] = round(float(majority), 4)
    metrics["beats_majority_baseline"] = bool(metrics["accuracy"] > majority)
    baseline = BASELINES.get(taxonomy, {})
    metrics["modernbert_baseline"] = baseline
    if baseline.get("accuracy") is not None:
        metrics["accuracy_delta_vs_modernbert"] = round(
            metrics["accuracy"] - baseline["accuracy"], 4)

    report = classification_report(df["true_label"], df["predicted_label"],
                                   output_dict=True, zero_division=0)
    coverage = coverage_curve(df)
    budget = accuracy_at_budget(df)

    (out_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(report).transpose().to_csv(out_dir / f"{prefix}_classification_report.csv")
    reliability.to_csv(out_dir / f"{prefix}_reliability.csv", index=False)
    coverage.to_csv(out_dir / f"{prefix}_threshold_sweep.csv", index=False)
    budget.to_csv(out_dir / f"{prefix}_accuracy_at_budget.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print("\nReliability (confidence vs actual accuracy):")
    print(reliability.to_string(index=False))
    print("\nAccuracy at fixed human-review budget:")
    print(budget.to_string(index=False))
    print(f"\nWrote 5 files to {out_dir}")


if __name__ == "__main__":
    main()
