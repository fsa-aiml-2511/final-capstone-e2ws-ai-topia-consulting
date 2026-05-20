#!/usr/bin/env python3
"""
Model 1: Traditional ML - Training Script
=========================================
XGBoost multiclass classifier for accident Severity 1, 2, 3, and 4.
"""
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from pipelines.Classification_pipelines import plot_feature_importance
from pipelines.data_cleaning_accident_pipeline import accident_engineer_features
from pipelines.data_pipeline import (
    clean_data,
    drop_low_variance_columns,
    get_data_and_process_target,
    load_raw_data,
    save_processed_data,
    scale_features,
    split_data,
)


TARGET = "Severity"
PROCESSED_FILENAME = "city_traffic_processed.csv"
PROCESSED_DATA = Path("data/processed")
SAVED_MODEL_DIR = Path("models/model1_traditional_ml/saved_model")

MODEL_FEATURES = [
    "is_weekend",
    "is_morning_rush",
    "is_evening_rush",
    "is_rush_hour",
    "duration_min",
    "Distance(mi)",
    "n_road_features",
    "has_traffic_control",
    "is_freezing",
    "low_visibility_severity",
    "has_precipitation",
    "weather_cluster_clear",
    "weather_cluster_cloudy",
    "weather_cluster_low_visibility",
    "weather_cluster_rain",
    "weather_cluster_snow_ice",
    "DangerousScore",
]


def load_data() -> pd.DataFrame:
    return load_raw_data("city_traffic_accidents.csv")


def processed_cache_is_usable(processed_path: Path) -> bool:
    if not processed_path.exists():
        return False

    probe = pd.read_csv(processed_path, nrows=1)
    required = {TARGET, "duration_min", "DangerousScore"}
    missing = required.difference(probe.columns)
    if missing:
        print(f"Regenerating processed data; cache is missing: {sorted(missing)}")
        processed_path.unlink()
        return False
    return True


def build_or_load_processed_data(df: pd.DataFrame) -> pd.DataFrame:
    processed_path = PROCESSED_DATA / PROCESSED_FILENAME

    if not processed_cache_is_usable(processed_path):
        df = clean_data(df)
        df = accident_engineer_features(df)
        df = drop_low_variance_columns(df)
        df = df.dropna(axis=1)

        available = [col for col in MODEL_FEATURES if col in df.columns]
        df = df[[TARGET] + available]
        save_processed_data(df, PROCESSED_FILENAME)
    else:
        print("Processed data found; loading from cache.")

    processed_df, target_stats = get_data_and_process_target(
        PROCESSED_FILENAME,
        target_column=TARGET,
    )
    if target_stats:
        print(f"\nReady to train Model 1 for {TARGET}.")

    return processed_df


def preprocess_features(df: pd.DataFrame):
    df = build_or_load_processed_data(df)

    X = df.drop(columns=[TARGET]).copy()
    y_raw = df[TARGET].astype(int)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    print("\nSeverity class distribution:")
    for encoded, label in enumerate(label_encoder.classes_):
        count = int((y == encoded).sum())
        print(f"  Severity {label}: {count:,} ({count / len(y):.1%})")

    X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2)
    X_train_scaled, X_val_scaled, scaler, feature_cols = scale_features(X_train, X_val)

    return X_train_scaled, X_val_scaled, y_train, y_val, scaler, label_encoder, feature_cols


def train_model(X_train, y_train, X_val, y_val):
    if len(X_train) > 80_000:
        from sklearn.model_selection import train_test_split

        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=80_000,
            random_state=42,
            stratify=y_train,
        )
        print(f"Subsampled to {len(y_train):,} rows for training.")

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    num_classes = int(np.unique(y_train).size)
    print(f"Training multiclass XGBoost with {num_classes} severity classes.")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    print(f"  Best iteration: {model.best_iteration}")
    return model


def evaluate_model(model, X_val, y_val, label_encoder) -> dict:
    proba = model.predict_proba(X_val)
    y_pred = np.argmax(proba, axis=1)
    target_names = [f"Severity {label}" for label in label_encoder.classes_]

    accuracy = accuracy_score(y_val, y_pred)
    weighted_f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

    print("\n--- Model Evaluation ---")
    print(f"Validation accuracy : {accuracy:.4f}")
    print(f"Validation F1 (wtd) : {weighted_f1:.4f}")
    print(classification_report(y_val, y_pred, target_names=target_names, zero_division=0))

    return {
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "classes": label_encoder.classes_.tolist(),
        "model_type": "multiclass_severity",
    }


def explain_model(model, X_val, y_val) -> None:
    print("\n--- Feature Importance Analysis ---")
    plot_feature_importance(model, X_val, y_val, "XGBoost")


def save_model(model, scaler, label_encoder, feature_cols, metrics) -> None:
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SAVED_MODEL_DIR / "model.joblib")
    joblib.dump(scaler, SAVED_MODEL_DIR / "scaler.joblib")
    joblib.dump(label_encoder, SAVED_MODEL_DIR / "label_encoder.joblib")
    joblib.dump(feature_cols, SAVED_MODEL_DIR / "feature_columns.joblib")
    joblib.dump(metrics, SAVED_MODEL_DIR / "metrics.joblib")
    print(f"Artifacts saved to {SAVED_MODEL_DIR}")


def main() -> None:
    df = load_data()
    X_train, X_val, y_train, y_val, scaler, label_encoder, feature_cols = preprocess_features(df)
    model = train_model(X_train, y_train, X_val, y_val)
    metrics = evaluate_model(model, X_val, y_val, label_encoder)
    save_model(model, scaler, label_encoder, feature_cols, metrics)
    explain_model(model, X_val, y_val)
    print("Training complete!")


if __name__ == "__main__":
    main()
