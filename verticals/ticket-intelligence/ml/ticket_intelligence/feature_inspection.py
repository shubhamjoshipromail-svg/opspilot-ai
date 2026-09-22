"""Inspect top TF-IDF logistic-regression features per class."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from data_utils import ARTIFACT_DIR, OUTPUT_DIR


def load_model(path: Path):
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def top_features(model, top_n: int = 25) -> pd.DataFrame:
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["clf"]
    feature_names = vectorizer.get_feature_names_out()
    rows: list[dict] = []

    if len(classifier.classes_) == 2 and classifier.coef_.shape[0] == 1:
        classes = classifier.classes_
        coefficients = classifier.coef_[0]
        for class_label, sign in [(classes[1], 1), (classes[0], -1)]:
            ranked = (coefficients * sign).argsort()[::-1][:top_n]
            for rank, idx in enumerate(ranked, start=1):
                rows.append(
                    {
                        "class": class_label,
                        "rank": rank,
                        "feature": feature_names[idx],
                        "weight": round(float(coefficients[idx] * sign), 6),
                    }
                )
        return pd.DataFrame(rows)

    for class_idx, class_label in enumerate(classifier.classes_):
        coefficients = classifier.coef_[class_idx]
        ranked = coefficients.argsort()[::-1][:top_n]
        for rank, idx in enumerate(ranked, start=1):
            rows.append(
                {
                    "class": class_label,
                    "rank": rank,
                    "feature": feature_names[idx],
                    "weight": round(float(coefficients[idx]), 6),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    category_model = load_model(ARTIFACT_DIR / "category_model.pkl")
    if category_model is not None:
        category_features = top_features(category_model)
        path = OUTPUT_DIR / "top_features_by_category.csv"
        category_features.to_csv(path, index=False)
        outputs["category"] = str(path)

    priority_model = load_model(ARTIFACT_DIR / "priority_model.pkl")
    if priority_model is not None:
        priority_features = top_features(priority_model)
        path = OUTPUT_DIR / "top_features_by_priority.csv"
        priority_features.to_csv(path, index=False)
        outputs["priority"] = str(path)

    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
