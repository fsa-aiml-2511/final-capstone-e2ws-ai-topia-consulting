#!/usr/bin/env python3
"""
Model 1: Traditional ML - Prediction Script
===========================================
Loads the trained XGBoost multiclass severity model and predicts accident
Severity 1, 2, 3, or 4 for new traffic accident records.

Output:
    test_data/model1_results.csv
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.data_pipeline import accident_predict_features


MODEL_DIR = PROJECT_ROOT / "models" / "model1_traditional_ml" / "saved_model"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FILE = TEST_DATA_DIR / "model1_results.csv"


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("model1_predict")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def load_artifacts(logger: logging.Logger):
    required = ["model.joblib", "scaler.joblib", "label_encoder.joblib", "feature_columns.joblib"]
    missing = [name for name in required if not (MODEL_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing Model 1 artifacts in {MODEL_DIR}: {missing}\n"
            "Run models/model1_traditional_ml/train.py first."
        )

    logger.info("Loading Model 1 artifacts from %s", MODEL_DIR)
    model = joblib.load(MODEL_DIR / "model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    feature_cols = joblib.load(MODEL_DIR / "feature_columns.joblib")

    if isinstance(label_encoder, dict):
        raise RuntimeError(
            "Model 1 artifacts are from the old binary High Risk/Standard Risk model. "
            "Run models/model1_traditional_ml/train.py to create the new 4-class severity artifacts."
        )

    if len(getattr(label_encoder, "classes_", [])) != 4:
        raise RuntimeError(
            "Model 1 label encoder does not contain four severity classes. "
            "Run models/model1_traditional_ml/train.py after updating the training data."
        )

    return model, scaler, label_encoder, feature_cols


def find_test_file() -> Path:
    result_names = {p.name for p in TEST_DATA_DIR.glob("model*_results.csv")}
    candidates = [
        p for p in TEST_DATA_DIR.glob("*.csv")
        if p.name not in result_names and not p.name.startswith(".")
    ]
    if not candidates:
        raise FileNotFoundError(f"No test CSV found in {TEST_DATA_DIR}")

    preferred = ["city_traffic_accidents_test.csv", "city_traffic_accidents.csv"]
    for name in preferred:
        match = next((p for p in candidates if p.name.lower() == name.lower()), None)
        if match:
            return match

    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def predict(df: pd.DataFrame, model, scaler, label_encoder, feature_cols: list) -> pd.DataFrame:
    processed_df = accident_predict_features(df)
    processed_df = processed_df.reindex(columns=feature_cols, fill_value=0)
    X_scaled = scaler.transform(processed_df)

    proba = model.predict_proba(X_scaled)
    pred_encoded = np.argmax(proba, axis=1)
    pred_labels = label_encoder.inverse_transform(pred_encoded)
    confidence = np.round(proba.max(axis=1), 4)

    id_col = next((col for col in df.columns if col.lower() == "id"), None)
    ids = df[id_col].values if id_col else np.arange(1, len(df) + 1)

    return pd.DataFrame(
        {
            "id": ids,
            "prediction": pred_labels,
            "probability": confidence,
            "confidence": confidence,
        }
    )


def main() -> None:
    logger = setup_logging()
    model, scaler, label_encoder, feature_cols = load_artifacts(logger)

    test_file = find_test_file()
    logger.info("Using test file: %s", test_file)
    test_df = pd.read_csv(test_file)
    logger.info("Loaded test data: %s", test_df.shape)

    results = predict(test_df, model, scaler, label_encoder, feature_cols)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)
    logger.info("Predictions saved to %s", OUTPUT_FILE)
    print(results.head())


if __name__ == "__main__":
    main()
