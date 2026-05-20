"""
Shared data pipeline helpers.

This module contains reusable loading, cleaning, splitting, scaling, reporting,
and accident-prediction feature logic used by the model scripts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw_data(filename: str) -> pd.DataFrame:
    """Load a CSV file from data/raw."""
    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            "Make sure the raw data has been downloaded to data/raw/."
        )
    return pd.read_csv(filepath, low_memory=False)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows and normalize object columns to lowercase strings."""
    df = df.copy().drop_duplicates()
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].str.lower()
    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create common date/time features when matching columns are present."""
    df = df.copy()

    for col in ["Start_Time", "End_Time", "created_date", "closed_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Start_Time" in df.columns:
        df["hour"] = df["Start_Time"].dt.hour
        df["day_of_week"] = df["Start_Time"].dt.dayofweek
        df["month"] = df["Start_Time"].dt.month

    if "day_of_week" in df.columns:
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    if "hour" in df.columns:
        df["is_morning_rush"] = df["hour"].between(7, 9).astype(int)
        df["is_evening_rush"] = df["hour"].between(16, 19).astype(int)
        df["is_rush_hour"] = (df["is_morning_rush"] | df["is_evening_rush"]).astype(int)

    if {"Start_Time", "End_Time"}.issubset(df.columns):
        df["duration_min"] = (
            (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60
        ).clip(0, 1440)

    if "created_date" in df.columns:
        df["created_hour"] = df["created_date"].dt.hour
        df["created_day_of_week"] = df["created_date"].dt.dayofweek
        df["created_month"] = df["created_date"].dt.month

    if "closed_date" in df.columns:
        df["closed_hour"] = df["closed_date"].dt.hour
        df["closed_day_of_week"] = df["closed_date"].dt.dayofweek
        df["closed_month"] = df["closed_date"].dt.month

    return df


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Split features and labels with stratification."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("--- Data Split Component ---")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_test.shape[0]} samples")
    print("\nTraining class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(f" Class {label}: {count} samples ({count / len(y_train):.1%})")

    return X_train, X_test, y_train, y_test


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """Save a processed CSV to data/processed."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")


def load_processed_data(filename: str) -> pd.DataFrame:
    """Load a processed CSV from data/processed."""
    filepath = PROCESSED_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data not found: {filepath}\n"
            "Run the data pipeline first to generate processed data."
        )
    return pd.read_csv(filepath)


def convert_bools_to_ints(df: pd.DataFrame) -> pd.DataFrame:
    """Convert boolean columns to 0/1 integers."""
    df = df.copy()
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def drop_low_variance_columns(df: pd.DataFrame, threshold: float = 0.99) -> pd.DataFrame:
    """Drop columns where one value occupies more than threshold of rows."""
    cols_to_drop = []
    for col in df.columns:
        counts = df[col].value_counts(normalize=True, dropna=False)
        if not counts.empty and counts.iloc[0] > threshold:
            cols_to_drop.append(col)

    print(f"Dropped {len(cols_to_drop)} columns with > {threshold:.0%} dominance.")
    if cols_to_drop:
        print(f"Columns dropped: {cols_to_drop}")
    return df.drop(columns=cols_to_drop)


def get_data_and_process_target(file_path: str, target_column: str):
    """Load a processed dataset and print simple target diagnostics."""
    df = load_processed_data(file_path)
    print("--- Data Successfully Loaded ---")
    print(f"Data shape: {df.shape}")

    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in dataframe.")
        return df, None

    stats = {
        "range": df[target_column].max() - df[target_column].min(),
        "std": df[target_column].std(),
    }
    print(f"Target Column: '{target_column}'")
    print(f"Target range: {stats['range']:,.2f}")
    print(f"Target std: {stats['std']:,.2f}")
    return df, stats


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Standardize feature columns using train-only fit."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    feature_cols = X_train.columns.tolist()
    print("--- Scaling Component ---")
    print(f"Scaler fitted on {len(feature_cols)} features.")
    return X_train_scaled, X_test_scaled, scaler, feature_cols


def label_encode_target(y):
    """Encode labels and return both encoded labels and the fitted encoder."""
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print("--- Label Encoding Component ---")
    for index, label in enumerate(label_encoder.classes_):
        print(f"  {label} -> {index}")
    return y_encoded, label_encoder


def print_model_report(y_test, y_test_pred, model_name: str) -> None:
    """Print a classification report and confusion matrix."""
    print("\n" + "=" * 60)
    print(f" MODEL: {model_name}")
    print("=" * 60)
    print(classification_report(y_test, y_test_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("=" * 60 + "\n")


def plot_feature_importance(trained_model, X_val, y_val, model_name: str, top_n: int = 30):
    """Calculate and plot permutation importances for a fitted model."""
    import matplotlib.pyplot as plt

    print(f"\nCalculating feature importance for {model_name}...")
    result = permutation_importance(
        trained_model,
        X_val,
        y_val,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    feature_names = (
        X_val.columns
        if isinstance(X_val, pd.DataFrame)
        else [f"Feature {index}" for index in range(X_val.shape[1])]
    )
    feat_imp = pd.DataFrame(
        {"feature": feature_names, "importance": result.importances_mean}
    ).sort_values("importance", ascending=False).head(top_n)

    print(f"Top {top_n} features for {model_name}:")
    for _, row in feat_imp.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    plt.figure(figsize=(10, 8))
    plt.barh(feat_imp["feature"][::-1], feat_imp["importance"][::-1], color="teal")
    plt.xlabel("Decrease in Accuracy (Permutation Importance)")
    plt.ylabel("Feature")
    plt.title(f"{model_name} - Top {top_n} Feature Importances")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    return feat_imp


def plot_prediction_probabilities(trained_model, X_test, model_name: str) -> None:
    """Plot max predicted probability distribution for classifiers."""
    import matplotlib.pyplot as plt

    if not hasattr(trained_model, "predict_proba"):
        print(f"{model_name} does not support predict_proba.")
        return

    probs = trained_model.predict_proba(X_test)
    max_probs = np.max(probs, axis=1)

    plt.figure(figsize=(10, 6))
    plt.hist(max_probs, bins=30, color="coral", edgecolor="black", alpha=0.7)
    plt.xlabel("Predicted Probability (Confidence)")
    plt.ylabel("Number of Samples")
    plt.title(f"Prediction Confidence Distribution - {model_name}")
    plt.grid(axis="y", alpha=0.3)
    plt.show()


def accident_predict_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prediction-safe accident feature engineering.

    This builds the same feature families Model 1 expects while avoiding dynamic
    operations that are unsafe on small or instructor-provided test sets.
    """
    from pipelines.data_cleaning_accident_pipeline import (
        dangerous_conditions_score,
        engineer_road_features,
        process_weather_features,
    )

    df = df.copy()
    df = df.drop(columns=["Country", "ID", "Source"], errors="ignore")

    for col in ["Start_Time", "End_Time", "Weather_Timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if {"Start_Time", "Weather_Timestamp"}.issubset(df.columns):
        df["Start_Time"] = df["Start_Time"].fillna(df["Weather_Timestamp"])

    if "Start_Time" in df.columns:
        df["hour"] = df["Start_Time"].dt.hour.fillna(12).astype(int)
        df["day_of_week"] = df["Start_Time"].dt.dayofweek.fillna(0).astype(int)
    else:
        df["hour"] = 12
        df["day_of_week"] = 0

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_morning_rush"] = df["hour"].between(7, 9).astype(int)
    df["is_evening_rush"] = df["hour"].between(16, 19).astype(int)
    df["is_rush_hour"] = (df["is_morning_rush"] | df["is_evening_rush"]).astype(int)

    if {"Start_Time", "End_Time"}.issubset(df.columns):
        df["duration_min"] = (
            (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60
        ).clip(0, 1440).fillna(0)
    else:
        df["duration_min"] = 0

    weather_numeric_cols = [
        "Temperature(F)",
        "Wind_Chill(F)",
        "Humidity(%)",
        "Pressure(in)",
        "Visibility(mi)",
        "Wind_Speed(mph)",
        "Precipitation(in)",
    ]
    for col in weather_numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median = df[col].median()
        df[col] = df[col].fillna(0.0 if pd.isna(median) else median)

    if "Weather_Condition" not in df.columns:
        df["Weather_Condition"] = "Clear"
    df["Weather_Condition"] = df["Weather_Condition"].fillna("Clear")

    for col in ["Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight"]:
        if col not in df.columns:
            df[col] = "Day"
        df[col] = df[col].fillna("Day")

    road_cols = [
        "Amenity",
        "Bump",
        "Crossing",
        "Give_Way",
        "Junction",
        "No_Exit",
        "Railway",
        "Roundabout",
        "Station",
        "Stop",
        "Traffic_Calming",
        "Traffic_Signal",
        "Turning_Loop",
    ]
    bool_map = {True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0}
    for col in road_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].map(bool_map).fillna(df[col]).fillna(0).astype(int)

    df = process_weather_features(df)
    df = dangerous_conditions_score(df)
    df = engineer_road_features(df)
    return df
