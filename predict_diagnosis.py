#!/usr/bin/env python3
"""Breast cancer prediction - inference only.

Loads the model saved by breast_cancer_detection.py (joblib) and scores tumour
measurements. No training happens here. Educational project, not for clinical use.

Usage:
    python predict_diagnosis.py                          (scores new_samples.csv)
    python predict_diagnosis.py new_samples.csv
    python predict_diagnosis.py new_samples.csv --screening
    python predict_diagnosis.py new_samples.csv --output scored_samples.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import joblib
import pandas as pd
import sklearn

LOGGER = logging.getLogger("diagnosis_predict")

DEFAULT_MODEL_PATH = Path("outputs/breast_cancer_model.joblib")
DEFAULT_METADATA_PATH = Path("outputs/model_metadata.json")
DEFAULT_INPUT_PATH = Path("new_samples.csv")


def load_metadata(path: Path) -> Dict[str, Any]:
    """Read the model card written during training."""
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}. Run breast_cancer_detection.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(path: Path):
    """Load the saved pipeline (imputer + scaler + classifier) from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Run breast_cancer_detection.py first.")
    return joblib.load(path)


def warn_if_version_differs(metadata: Dict[str, Any]) -> None:
    """A model should be loaded with the scikit-learn version that built it."""
    trained_with = metadata.get("scikit_learn_version")
    if trained_with and trained_with != sklearn.__version__:
        LOGGER.warning("Model was trained with scikit-learn %s but %s is installed. "
                       "Results may be wrong or loading may fail.", trained_with, sklearn.__version__)


def read_samples(path: Path, metadata: Dict[str, Any]) -> pd.DataFrame:
    """Read samples from CSV and check they match what the model expects."""
    if not path.exists():
        raise FileNotFoundError(f"Sample file not found: {path}")

    data = pd.read_csv(path)
    features = metadata["features"]

    missing_columns = [c for c in features if c not in data.columns]
    if missing_columns:
        raise ValueError(f"Input is missing required columns: {missing_columns}")

    for column in features:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"Column '{column}' must contain numbers only")

    empty_rows = data.index[data[features].isnull().any(axis=1)].tolist()
    if empty_rows:
        raise ValueError(f"Rows with missing values (0-based row numbers): {empty_rows}")

    outside = {}
    for column, (low, high) in metadata["training_ranges"].items():
        rows = data.index[(data[column] < low) | (data[column] > high)].tolist()
        if rows:
            outside[column] = rows
    if outside:
        LOGGER.warning("Values outside the training range (treat these results with care): %s", outside)
    return data


def score_samples(model, data: pd.DataFrame, metadata: Dict[str, Any], threshold: float) -> pd.DataFrame:
    """Add malignancy probability and a Benign/Malignant prediction."""
    probability = model.predict_proba(data[metadata["features"]])[:, 1]
    result = pd.DataFrame({
        "MalignancyProbability": probability.round(3),
        "Prediction": ["Malignant" if p >= threshold else "Benign" for p in probability],
    }, index=data.index)
    if "Actual" in data.columns:
        result["Actual"] = data["Actual"]
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict_diagnosis.py",
        description="Score tumour measurements with the saved model (no retraining). Not for clinical use.")
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT_PATH,
                        help=f"CSV file with the 30 measurement columns (default: {DEFAULT_INPUT_PATH})")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="joblib model file")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH, help="model metadata JSON")
    parser.add_argument("--screening", action="store_true",
                        help="use the lower screening threshold stored with the model "
                             "(catches more malignant cases, creates more false alarms)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="custom probability threshold between 0 and 1 (overrides --screening)")
    parser.add_argument("--output", type=Path, default=None, help="save scored samples to CSV")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    try:
        metadata = load_metadata(args.metadata)
        warn_if_version_differs(metadata)
        model = load_model(args.model)
        data = read_samples(args.input, metadata)
    except (OSError, ValueError, KeyError) as error:
        LOGGER.error("%s", error)
        return 1

    if args.threshold is not None:
        if not 0.0 < args.threshold < 1.0:
            LOGGER.error("--threshold must be between 0 and 1")
            return 1
        threshold = args.threshold
    elif args.screening:
        threshold = metadata["screening_threshold"]
    else:
        threshold = metadata["decision_threshold"]

    result = score_samples(model, data, metadata, threshold)
    LOGGER.info("Scored %d sample(s) using threshold %.2f", len(result), threshold)
    print(result.to_string())
    print("\nNot for clinical use. Educational project.")

    if args.output:
        result.to_csv(args.output, index=False)
        LOGGER.info("Saved scored samples to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
