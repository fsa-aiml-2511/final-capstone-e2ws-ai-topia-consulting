#!/usr/bin/env python3
"""
Model 5: Innovation — Training Script
=====================================

Urban Complaint Response Optimizer (Unsupervised / Hybrid Scoring)

Value proposition
-----------------
This model helps a city triage 311 complaints faster by combining:
1) unsupervised text clustering (TF-IDF + KMeans), and
2) domain-informed urgency signals from complaint text.

Why this matters:
- Pure rules miss latent patterns in complaint language.
- Pure clustering is hard to interpret operationally.
- This hybrid design keeps the model unsupervised while producing an
  interpretable urgency score for response prioritization.

Success metrics
---------------
Because this is an unsupervised model, we evaluate it with:
- Silhouette score: separation/cohesion of clusters
- Calinski-Harabasz score: cluster compactness and separation
- Davies-Bouldin score: lower is better
- Priority lift: whether high-priority complaints have meaningfully higher
  urgency scores than normal complaints
- Proxy tier agreement: how well cluster-derived severity lines up with
  domain proxy tiers created from emergency/distress/moderate keyword signals

Artifacts saved
---------------
- models/model5_innovation/saved_model/model.joblib
- models/model5_innovation/saved_model/metrics.json
- models/model5_innovation/saved_model/training_scored_sample.csv

to run
python -u models/model5_innovation/train.py
"""
from __future__ import annotations

import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path.cwd()
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "urbanpulse_311_complaints.csv"
RAW_DATA = PROJECT_ROOT / "data" / "raw"
SAVED_MODEL_DIR = PROJECT_ROOT / "models" / "model5_innovation" / "saved_model"
RANDOM_STATE = 42

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
TEXT_COLS = ["descriptor", "resolution_description"]
PRIMARY_ID_COL = "unique_key"
DEFAULT_N_CLUSTERS = 3
TFIDF_MAX_FEATURES = 5000

# ---------------------------------------------------------------------
# Keyword dictionaries
# ---------------------------------------------------------------------
URGENT_PATTERNS = [
    r"\bfire\b", r"\bflood\b", r"\bflooding\b", r"\bgas leak\b", r"\bleak\b",
    r"\bsmoke\b", r"\bexplosion\b", r"\bunsafe\b", r"\bdanger\b", r"\bhazard\b",
    r"\bhazardous\b", r"\binjury\b", r"\binjured\b", r"\bblood\b",
    r"\baccident\b", r"\bcrash\b", r"\bcollapse\b", r"\bsinkhole\b",
    r"\bblocked\b", r"\bno heat\b", r"\bno water\b", r"\bsewage\b",
    r"\blive wire\b", r"\belectrical\b", r"\bpower outage\b",
    r"\bemergency\b", r"\burgent\b", r"\bimmediately\b", r"\basap\b",
    r"\bchild\b", r"\belderly\b", r"\bdisabled\b", r"\bwheelchair\b",
    r"\bmedical\b", r"\bambulance\b", r"\b911\b", r"\bviolence\b",
    r"\bassault\b", r"\bthreat\b", r"\bweapon\b",
]

DISTRESS_PATTERNS = [
    r"\bhelp\b", r"\bplease help\b", r"\bdesperate\b", r"\bcrying\b",
    r"\bscared\b", r"\bterrified\b", r"\bcannot breathe\b", r"\bpanic\b",
    r"\bstuck\b", r"\btrapped\b", r"\bstranded\b", r"\bno response\b",
    r"\bwaiting for hours\b", r"\bkids\b", r"\bbaby\b", r"\bgrandma\b",
    r"\bgrandfather\b", r"\bsick\b", r"\bunsafe for family\b",
]

MODERATE_PATTERNS = [
    r"\brepair\b", r"\bbroken\b", r"\bcracked\b", r"\bpothole\b",
    r"\btrash\b", r"\bgarbage\b", r"\bnoise\b", r"\bparking\b",
    r"\bstreet light\b", r"\blight out\b", r"\bsign missing\b",
    r"\bwater leak\b", r"\bmold\b", r"\bdrain\b", r"\bstanding water\b",
    r"\broad damage\b", r"\bsidewalk\b", r"\btraffic signal\b",
    r"\bmissed pickup\b", r"\brodent\b", r"\binfestation\b",
]

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("model5_innovation_train")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


LOGGER = setup_logging()


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Basic text normalization used in the notebook."""
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\-/&]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_pattern_hits(text: str, patterns: List[str]) -> int:
    """Count keyword-pattern matches in text."""
    if not isinstance(text, str) or text.strip() == "":
        return 0
    text = text.lower()
    return sum(int(bool(re.search(pattern, text))) for pattern in patterns)


def safe_json(value):
    """Convert numpy/pandas scalars to JSON-safe Python types."""
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def find_input_csv() -> Path:
    """
    Find the best available training CSV.

    Preference order:
    1) data/processed/*.csv
    2) data/raw/*.csv
    """

    return DATA_PATH

    raise FileNotFoundError(
        "No training CSV found. Expected a file in data/processed/ or data/raw/."
    )


def build_proxy_priority(row: pd.Series) -> Tuple[int, str]:
    """
    Build a proxy operational priority from rule signals.

    This is NOT a supervised label. It is only a domain-aligned benchmark
    used to validate whether the unsupervised outputs make operational sense.
    """
    if row["urgent_keyword_count"] > 0 or row["distress_keyword_count"] > 0:
        return 1, "urgent"
    if row["moderate_keyword_count"] > 0:
        return 2, "elevated"
    return 3, "normal"


def compute_score_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create text-derived numeric signals used in the final urgency score."""
    out = df.copy()
    out["urgent_keyword_count"] = out["complaint_text"].apply(
        lambda x: count_pattern_hits(x, URGENT_PATTERNS)
    )
    out["distress_keyword_count"] = out["complaint_text"].apply(
        lambda x: count_pattern_hits(x, DISTRESS_PATTERNS)
    )
    out["moderate_keyword_count"] = out["complaint_text"].apply(
        lambda x: count_pattern_hits(x, MODERATE_PATTERNS)
    )
    out["text_length"] = out["complaint_text"].str.len().fillna(0)
    out["exclamation_count"] = out["complaint_text"].str.count("!").fillna(0)
    out["all_caps_word_count"] = out["complaint_text"].apply(
        lambda x: len(re.findall(r"\b[A-Z]{3,}\b", x)) if isinstance(x, str) else 0
    )

    priority_info = out.apply(build_proxy_priority, axis=1, result_type="expand")
    out["proxy_priority"] = priority_info[0]
    out["proxy_tier"] = priority_info[1]
    return out


# ---------------------------------------------------------------------
# Main pipeline functions
# ---------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load source dataset from processed or raw data directories."""
    data_path = find_input_csv()
    LOGGER.info("Loading training data from %s", data_path)
    df = pd.read_csv(data_path)
    LOGGER.info("Loaded %s rows and %s columns", df.shape[0], df.shape[1])

    # Create missing text columns if needed
    for col in TEXT_COLS:
        if col not in df.columns:
            LOGGER.warning("Missing column '%s'; creating empty fallback", col)
            df[col] = ""

    if PRIMARY_ID_COL not in df.columns:
        LOGGER.warning("Missing id column '%s'; generating synthetic ids", PRIMARY_ID_COL)
        df[PRIMARY_ID_COL] = np.arange(1, len(df) + 1)

    return df


def preprocess(df: pd.DataFrame):
    """
    Prepare text and proxy signals.

    Returns:
        processed dataframe
        fitted TF-IDF vectorizer
        TF-IDF matrix
    """
    work = df.copy()

    for col in TEXT_COLS:
        work[col] = work[col].fillna("")

    LOGGER.info("Building complaint_text field")
    work["complaint_text"] = (
        work["descriptor"].map(clean_text)
        + " | "
        + work["resolution_description"].map(clean_text)
    ).str.strip(" |")

    work = work[work["complaint_text"].str.strip() != ""].copy()
    work = work.reset_index(drop=True)

    work = compute_score_features(work)

    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=3,
    )
    X_tfidf = tfidf.fit_transform(work["complaint_text"])
    LOGGER.info("TF-IDF matrix shape: %s", X_tfidf.shape)

    return work, tfidf, X_tfidf


def train_model(df: pd.DataFrame, X_tfidf):
    """
    Train unsupervised text clustering model and derive urgency score.
    """
    LOGGER.info("Training KMeans clustering model with %s clusters", DEFAULT_N_CLUSTERS)
    kmeans = KMeans(
        n_clusters=DEFAULT_N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=20,
    )
    df = df.copy()
    df["cluster_id"] = kmeans.fit_predict(X_tfidf)

    cluster_profile = (
        df.groupby("cluster_id")[["urgent_keyword_count", "distress_keyword_count", "moderate_keyword_count"]]
        .mean()
        .reset_index()
    )

    cluster_profile["cluster_severity_score"] = (
        2.0 * cluster_profile["urgent_keyword_count"]
        + 1.5 * cluster_profile["distress_keyword_count"]
        + 1.0 * cluster_profile["moderate_keyword_count"]
    )
    cluster_profile = cluster_profile.sort_values("cluster_severity_score").reset_index(drop=True)

    # Lowest severity cluster -> 0, middle -> 1, highest -> 2
    cluster_rank_map = {
        safe_json(cluster_profile.loc[i, "cluster_id"]): i
        for i in range(len(cluster_profile))
    }
    df["cluster_severity_rank"] = df["cluster_id"].map(cluster_rank_map)

    score_cols = [
        "urgent_keyword_count",
        "moderate_keyword_count",
        "distress_keyword_count",
        "text_length",
        "exclamation_count",
        "all_caps_word_count",
        "cluster_severity_rank",
    ]

    scaler = MinMaxScaler()
    scaled_cols = [f"{c}_scaled" for c in score_cols]
    df[scaled_cols] = scaler.fit_transform(df[score_cols])

    df["urgency_score"] = (
        0.40 * df["urgent_keyword_count_scaled"]
        + 0.13 * df["moderate_keyword_count_scaled"]
        + 0.27 * df["distress_keyword_count_scaled"]
        + 0.13 * df["cluster_severity_rank_scaled"]
        + 0.04 * df["exclamation_count_scaled"]
        + 0.03 * df["all_caps_word_count_scaled"]
    )

    # Explicit emergency-language uplift
    df["urgency_score"] = np.where(
        df["urgent_keyword_count"] > 0,
        df["urgency_score"] + 0.10,
        df["urgency_score"],
    ).clip(0, 1)

    # Final operational tier
    df["urgency_tier"] = np.select(
        [
            (df["urgency_score"] >= 0.30) | (df["proxy_priority"] == 1),
            (df["urgency_score"] >= 0.07) | (df["proxy_priority"] == 2),
        ],
        [
            "urgent",
            "elevated",
        ],
        default="normal",
    )

    tier_priority_map = {"urgent": 1, "elevated": 2, "normal": 3}
    df["response_priority"] = df["urgency_tier"].map(tier_priority_map).astype(int)

    model_bundle = {
        "tfidf_vectorizer": None,  # assigned later in main
        "kmeans_model": kmeans,
        "scaler": scaler,
        "score_columns": score_cols,
        "scaled_columns": scaled_cols,
        "cluster_profile": cluster_profile.to_dict(orient="records"),
        "cluster_rank_map": {int(k): int(v) for k, v in cluster_rank_map.items()},
        "tier_priority_map": tier_priority_map,
        "config": {
            "n_clusters": DEFAULT_N_CLUSTERS,
            "random_state": RANDOM_STATE,
            "tfidf_max_features": TFIDF_MAX_FEATURES,
            "text_columns": TEXT_COLS,
            "id_column": PRIMARY_ID_COL,
        },
    }

    return df, model_bundle


def evaluate_model(df: pd.DataFrame, X_tfidf, kmeans) -> Dict:
    """
    Evaluate with unsupervised and business-alignment metrics.

    Why these metrics:
    - Silhouette / Calinski-Harabasz / Davies-Bouldin are standard internal
      clustering metrics for unsupervised models.
    - Proxy-tier agreement checks whether learned clusters align with
      domain-informed urgency structure without turning the task into
      supervised classification.
    - Priority lift estimates operational ROI by checking how much more
      concentrated urgent-score mass is in high-priority cases.
    """
    LOGGER.info("Evaluating unsupervised model")

    labels = df["cluster_id"].values
    n_rows = len(df)

    if len(np.unique(labels)) > 1 and n_rows > len(np.unique(labels)):
        sil = float(silhouette_score(X_tfidf, labels, sample_size=min(5000, n_rows), random_state=RANDOM_STATE))
        dense_X = X_tfidf.toarray()
        ch = float(calinski_harabasz_score(dense_X, labels))
        db = float(davies_bouldin_score(dense_X, labels))
    else:
        sil, ch, db = None, None, None

    proxy_code_map = {"normal": 0, "elevated": 1, "urgent": 2}
    proxy_codes = df["proxy_tier"].map(proxy_code_map).values
    pred_codes = df["urgency_tier"].map(proxy_code_map).values

    # Agreement with domain proxy tiers (not supervised accuracy; just alignment)
    tier_match_rate = float(np.mean(proxy_codes == pred_codes))
    nmi_proxy = float(normalized_mutual_info_score(proxy_codes, labels))
    ari_proxy = float(adjusted_rand_score(proxy_codes, labels))

    mean_scores = df.groupby("urgency_tier")["urgency_score"].mean().to_dict()
    tier_distribution = df["urgency_tier"].value_counts(normalize=True).sort_index().to_dict()
    cluster_distribution = df["cluster_id"].value_counts(normalize=True).sort_index().to_dict()

    urgent_mean = float(mean_scores.get("urgent", 0.0))
    normal_mean = float(mean_scores.get("normal", 0.0))
    elevated_mean = float(mean_scores.get("elevated", 0.0))

    priority_lift_vs_normal = (
        urgent_mean / normal_mean if normal_mean and normal_mean > 0 else None
    )
    elevated_lift_vs_normal = (
        elevated_mean / normal_mean if normal_mean and normal_mean > 0 else None
    )

    top_decile = max(1, int(0.10 * len(df)))
    top10 = df.nlargest(top_decile, "urgency_score")
    urgent_capture_top10 = float((top10["proxy_tier"] == "urgent").mean())
    baseline_urgent_rate = float((df["proxy_tier"] == "urgent").mean())
    urgent_capture_lift_top10 = (
        urgent_capture_top10 / baseline_urgent_rate if baseline_urgent_rate > 0 else None
    )

    business_impact = {
        "baseline_urgent_rate": baseline_urgent_rate,
        "urgent_rate_in_top_10pct_scored_cases": urgent_capture_top10,
        "top_10pct_urgent_capture_lift": urgent_capture_lift_top10,
        "mean_urgency_score_by_tier": {k: float(v) for k, v in mean_scores.items()},
        "priority_lift_vs_normal": priority_lift_vs_normal,
        "elevated_lift_vs_normal": elevated_lift_vs_normal,
        "interpretation": (
            "A useful triage model should rank urgent-like complaints much higher than "
            "normal complaints. Lift > 1 means scarce response resources are being "
            "focused on more severe complaints."
        ),
    }

    metrics = {
        "model_type": "unsupervised_hybrid_tfidf_kmeans_scoring",
        "n_rows_used_for_training": int(len(df)),
        "n_clusters": int(kmeans.n_clusters),
        "silhouette_score": sil,
        "calinski_harabasz_score": ch,
        "davies_bouldin_score": db,
        "proxy_tier_match_rate": tier_match_rate,
        "normalized_mutual_info_vs_proxy_tiers": nmi_proxy,
        "adjusted_rand_index_vs_proxy_tiers": ari_proxy,
        "tier_distribution": {str(k): float(v) for k, v in tier_distribution.items()},
        "cluster_distribution": {str(k): float(v) for k, v in cluster_distribution.items()},
        "business_impact": business_impact,
        "success_metric_explanation": {
            "primary_internal_metric": "silhouette_score",
            "primary_business_metric": "top_10pct_urgent_capture_lift",
            "why": (
                "Silhouette evaluates clustering quality without labels. "
                "Urgent-capture lift estimates whether the scoring system helps "
                "operations surface more urgent complaints near the top of the queue."
            ),
        },
    }

    LOGGER.info("Silhouette score: %s", metrics["silhouette_score"])
    LOGGER.info("Proxy tier match rate: %.4f", metrics["proxy_tier_match_rate"])
    LOGGER.info(
        "Top 10%% urgent capture lift: %s",
        metrics["business_impact"]["top_10pct_urgent_capture_lift"],
    )

    return metrics


def save_model(model_bundle: Dict, metrics: Dict, scored_df: pd.DataFrame) -> None:
    """Save model artifacts to saved_model/."""
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = SAVED_MODEL_DIR / "model.joblib"
    metrics_path = SAVED_MODEL_DIR / "metrics.json"
    sample_path = SAVED_MODEL_DIR / "training_scored_sample.csv"

    joblib.dump(model_bundle, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=safe_json)

    cols_to_save = [
        PRIMARY_ID_COL,
        "complaint_text",
        "cluster_id",
        "cluster_severity_rank",
        "proxy_tier",
        "urgency_tier",
        "response_priority",
        "urgency_score",
    ]
    available_cols = [c for c in cols_to_save if c in scored_df.columns]
    scored_df[available_cols].head(5000).to_csv(sample_path, index=False)

    LOGGER.info("Saved model to %s", model_path)
    LOGGER.info("Saved metrics to %s", metrics_path)
    LOGGER.info("Saved scored sample to %s", sample_path)


def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # 1. Load data
    df = load_data()

    # 2. Preprocess
    processed_df, tfidf_vectorizer, X_tfidf = preprocess(df)

    # 3. Train
    scored_df, model_bundle = train_model(processed_df, X_tfidf)
    model_bundle["tfidf_vectorizer"] = tfidf_vectorizer

    # 4. Evaluate
    metrics = evaluate_model(
        scored_df,
        X_tfidf,
        model_bundle["kmeans_model"],
    )

    # 5. Save
    save_model(model_bundle, metrics, scored_df)

    LOGGER.info("Training complete!")


if __name__ == "__main__":
    main()
