#!/usr/bin/env python3
"""
Model 4: NLP Complaint Routing — Training Script
=================================================
Trains two SGDClassifier pipelines on 311 complaint text:
  1. Routing classifier  — predicts responsible agency (HPD, NYPD, DSNY, ...)
  2. Category classifier — maps complaint to 6 simplified buckets

Both pipelines use:
  char-level TF-IDF + word-level TF-IDF + ExtraTextFeatures (10 keyword signals)

ExtraTextFeatures MUST be identical here, in predict.py, and in webapp/app.py.
If they differ, sklearn will raise a feature-count mismatch at inference time.

To run from project root:
    python -u models/model4_nlp_classification/train.py
"""

from __future__ import annotations

import logging
import re
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

PROJECT_ROOT    = Path(__file__).resolve().parents[2]
DATA_PATH       = PROJECT_ROOT / "data" / "raw" / "urbanpulse_311_complaints.csv"
SAVED_MODEL_DIR = PROJECT_ROOT / "models" / "model4_nlp_classification" / "saved_model"
SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Agencies to include — must match the routing label encoder classes
KEEP_AGENCIES = {
    "DCWP", "DEP", "DHS", "DOB", "DOE", "DOHMH",
    "DOT", "DPR", "DSNY", "HPD", "NYPD", "OOS", "OTI", "TLC",
}

# 6-bucket category mapping — order matters (first match wins)
CATEGORY_PATTERNS = [
    ("blocked driveway",    r"blocked.driveway|driveway"),
    ("heat/hot water",      r"\bheat\b|\bhot water\b|\bheating\b|\bboiler\b|\bno heat\b"),
    ("illegal parking",     r"illegal.parking|\bparking\b"),
    ("noise - residential", r"\bnoise\b"),
    ("snow or ice",         r"\bsnow\b|\bice\b|\bicy\b|\bslippery\b"),
]


# ===========================================================================
# ExtraTextFeatures — MUST match predict.py and webapp/app.py exactly
# ===========================================================================
class ExtraTextFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        s = pd.Series(X).fillna("").astype(str).str.lower()
        extra = pd.DataFrame({
            "has_snow_or_ice":    s.str.contains(r"\bsnow\b|\bice\b|\bicy\b|\bslush\b", regex=True).astype(int),
            "has_sanitation":     s.str.contains(r"\btrash\b|\bgarbage\b|\brecycling\b|\bsanitation\b|\bdumping\b", regex=True).astype(int),
            "has_driveway":       s.str.contains(r"\bdriveway\b", regex=True).astype(int),
            "has_blocked":        s.str.contains(r"\bblocked\b|\bblocking\b|\bobstructing\b", regex=True).astype(int),
            "has_parking":        s.str.contains(r"\bparking\b|\bparked\b|\bvehicle\b|\bcar\b|\btruck\b|\bdouble parked\b|\bhydrant\b", regex=True).astype(int),
            "has_noise":          s.str.contains(r"\bnoise\b|\bloud\b|\bmusic\b|\bbanging\b|\byelling\b|\bparty\b", regex=True).astype(int),
            "has_heat_hot_water": s.str.contains(r"\bno heat\b|\bheat\b|\bheating\b|\bhot water\b|\bboiler\b|\bradiator\b", regex=True).astype(int),
            "has_housing":        s.str.contains(r"\bapartment\b|\btenant\b|\blandlord\b|\bresidential\b", regex=True).astype(int),
            "has_dob_signal":     s.str.contains(r"\bconstruction\b|\bpermit\b|\bscaffold\b|\bdemolition\b|\bunsafe construction\b", regex=True).astype(int),
            "has_oos_signal":     s.str.contains(r"\bsheriff\b|\bmarshal\b|\beviction\b|\blockout\b|\bcivil enforcement\b", regex=True).astype(int),
        })
        return csr_matrix(extra.values)


# ===========================================================================
# Helpers
# ===========================================================================
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("model4_train")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"[^\w\s\-/&]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_routing_text(df: pd.DataFrame) -> pd.Series:
    """Same concatenation used in predict.py — must stay in sync."""
    ct  = df["complaint_type"].fillna("").map(clean_text)
    dsc = df["descriptor"].fillna("").map(clean_text)
    res = df["resolution_description"].fillna("").map(clean_text) if "resolution_description" in df.columns else pd.Series([""] * len(df), index=df.index)
    return (ct + " | " + dsc + " | " + res).str.strip(" |")


def map_category(complaint_type: str) -> str:
    if not isinstance(complaint_type, str):
        return "other"
    ct = complaint_type.lower()
    for label, pattern in CATEGORY_PATTERNS:
        if re.search(pattern, ct):
            return label
    return "other"


def build_pipeline(max_word: int = 12_000, max_char: int = 5_000) -> Pipeline:
    return Pipeline([
        ("features", FeatureUnion([
            ("word_tfidf", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 3),
                max_features=max_word,
                min_df=10,
                max_df=0.95,
                sublinear_tf=True,
            )),
            ("char_tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 4),
                max_features=max_char,
                min_df=10,
                max_df=0.95,
                sublinear_tf=True,
            )),
            ("extra_features", ExtraTextFeatures()),
        ])),
        ("clf", SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            max_iter=20,
            tol=1e-3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def evaluate(name: str, pipeline: Pipeline, X_test, y_test, le: LabelEncoder) -> None:
    y_pred = pipeline.predict(X_test)
    print(f"\n=== {name} Evaluation ===")
    print(f"Accuracy:             {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision (weighted): {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall (weighted):    {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1 (weighted):        {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))


# ===========================================================================
# Main
# ===========================================================================
def main():
    np.random.seed(42)
    logger = setup_logging()

    logger.info("Loading data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    logger.info("Loaded shape: %s", df.shape)

    # Ensure required columns exist
    for col in ("descriptor", "resolution_description"):
        if col not in df.columns:
            df[col] = ""

    # Keep only known agencies
    df = df[df["agency"].isin(KEEP_AGENCIES)].copy()
    logger.info("After agency filter: %d rows", len(df))

    # Build shared text feature
    df["complaint_text"] = build_routing_text(df)
    df = df[df["complaint_text"].str.strip() != ""].reset_index(drop=True)

    # Build category labels
    df["complaint_category"] = df["complaint_type"].apply(map_category)

    logger.info("Complaint category distribution:\n%s", df["complaint_category"].value_counts().to_string())
    print("\nLabel mapping (routing):", df["agency"].value_counts().index.tolist())

    # ── Category classifier ───────────────────────────────────────────────────
    label_encoder = LabelEncoder()
    y_cat = label_encoder.fit_transform(df["complaint_category"])
    logger.info("Category classes: %s", label_encoder.classes_.tolist())

    X_tr_c, X_va_c, y_tr_c, y_va_c = train_test_split(
        df["complaint_text"], y_cat,
        test_size=0.20, random_state=42, stratify=y_cat,
    )

    logger.info("Training category classifier on %d samples …", len(X_tr_c))
    model_pipeline = build_pipeline(max_word=12_000, max_char=5_000)
    model_pipeline.fit(X_tr_c, y_tr_c)
    evaluate("Category classifier", model_pipeline, X_va_c, y_va_c, label_encoder)

    # ── Routing classifier ────────────────────────────────────────────────────
    label_encoder_rt = LabelEncoder()
    y_rt = label_encoder_rt.fit_transform(df["agency"])
    logger.info("Routing classes: %s", label_encoder_rt.classes_.tolist())

    X_tr_r, X_va_r, y_tr_r, y_va_r = train_test_split(
        df["complaint_text"], y_rt,
        test_size=0.20, random_state=42, stratify=y_rt,
    )

    logger.info("Training routing classifier on %d samples …", len(X_tr_r))
    model_pipeline_rt = build_pipeline(max_word=12_000, max_char=5_000)
    model_pipeline_rt.fit(X_tr_r, y_tr_r)
    evaluate("Routing classifier", model_pipeline_rt, X_va_r, y_va_r, label_encoder_rt)

    # ── Save artifacts ────────────────────────────────────────────────────────
    logger.info("Saving artifacts to %s", SAVED_MODEL_DIR)
    joblib.dump(model_pipeline,    SAVED_MODEL_DIR / "model4_category_classifier_char_tfidf_SGD.pkl")
    joblib.dump(model_pipeline_rt, SAVED_MODEL_DIR / "model4_routing_classifier_char_tfidf_SGD.pkl")
    joblib.dump(label_encoder,     SAVED_MODEL_DIR / "model4_category_label_encoder.pkl")
    joblib.dump(label_encoder_rt,  SAVED_MODEL_DIR / "model4_routing_label_encoder.pkl")
    logger.info("All artifacts saved.")
    print(f"\nSaved artifacts to: {SAVED_MODEL_DIR}")


if __name__ == "__main__":
    main()
