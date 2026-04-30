#!/usr/bin/env python3
"""
Model 3: Pothole Detection — EfficientNetB0 Transfer Learning
==============================================================
Binary image classification: pothole (1) vs. normal road (0).

Dataset layout expected:
    data/raw/pothole_images/positive/   ← pothole images
    data/raw/pothole_images/negative/   ← normal road images

Two-phase training from notebook:
  Phase 1 — frozen backbone, train head (15 epochs)
  Phase 2 — unfreeze last 30 layers, fine-tune (10 epochs, lr=1e-5)

Best classification threshold is tuned on the validation set.

To run from project root:
    python -u models/model3_cnn/train.py
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

PROJECT_ROOT    = Path.cwd()
POS_FOLDER      = PROJECT_ROOT / "data" / "raw" / "pothole_images" / "pothole_images" / "positive"
NEG_FOLDER      = PROJECT_ROOT / "data" / "raw" / "pothole_images" / "pothole_images" / "negative"
SAVED_MODEL_DIR = PROJECT_ROOT / "models" / "model3_cnn" / "saved_model"
SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE   = 384
BATCH_SIZE = 32
CLASS_WEIGHT = {0: 1.0, 1: 2.0}


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("model3_train")
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


def build_dataframe() -> pd.DataFrame:
    rows = []
    for folder, label in [(POS_FOLDER, 1), (NEG_FOLDER, 0)]:
        for f in os.listdir(folder):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append({"filepath": str(folder / f), "label": label})
    return pd.DataFrame(rows)


def make_dataset(dataframe: pd.DataFrame, training: bool = False):
    import tensorflow as tf
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    AUTOTUNE = tf.data.AUTOTUNE

    def crop_road_region(img):
        h = tf.shape(img)[0]
        w = tf.shape(img)[1]
        y1 = tf.cast(tf.cast(h, tf.float32) * 0.30, tf.int32)
        y2 = tf.cast(tf.cast(h, tf.float32) * 0.78, tf.int32)
        x1 = tf.cast(tf.cast(w, tf.float32) * 0.05, tf.int32)
        x2 = tf.cast(tf.cast(w, tf.float32) * 0.95, tf.int32)
        return img[y1:y2, x1:x2, :]

    def load_image(filepath, label):
        img = tf.io.read_file(filepath)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = crop_road_region(img)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = preprocess_input(img * 255.0)
        return img, tf.cast(label, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices(
        (dataframe["filepath"].values, dataframe["label"].values)
    )
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(256, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def build_model():
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        tf.keras.layers.RandomContrast(0.20),
        tf.keras.layers.RandomBrightness(0.15),
        tf.keras.layers.GaussianNoise(0.03),
    ], name="augmentation")

    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs  = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = data_augmentation(inputs)
    x       = base_model(x, training=False)
    x       = tf.keras.layers.GlobalAveragePooling2D()(x)
    x       = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model, base_model


def train_model(model, base_model, train_ds, val_ds):
    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
    ]

    print("\nPhase 1 — training head (backbone frozen)")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        class_weight=CLASS_WEIGHT,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nPhase 2 — fine-tuning last 30 backbone layers")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        class_weight=CLASS_WEIGHT,
        callbacks=callbacks,
        verbose=1,
    )
    return history1, history2


def find_best_threshold(model, val_ds, val_df: pd.DataFrame) -> float:
    probs = model.predict(val_ds).ravel()
    true  = val_df["label"].values
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.20, 0.81, 0.02):
        preds = (probs >= t).astype(int)
        wf1   = f1_score(true, preds, average="weighted")
        if wf1 > best_f1:
            best_f1, best_t = wf1, float(t)
    return best_t


def main():
    logger = setup_logging()

    for folder in [POS_FOLDER, NEG_FOLDER]:
        if not folder.exists():
            logger.error("Missing image folder: %s", folder)
            sys.exit(1)

    logger.info("Building dataframe")
    df = build_dataframe()
    logger.info("Class distribution:\n%s", df["label"].value_counts().to_string())

    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
    val_df,  test_df  = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42)
    logger.info("Split — train:%d  val:%d  test:%d", len(train_df), len(val_df), len(test_df))

    train_ds = make_dataset(train_df, training=True)
    val_ds   = make_dataset(val_df,  training=False)
    test_ds  = make_dataset(test_df, training=False)

    logger.info("Building EfficientNetB0 model (IMG_SIZE=%d)", IMG_SIZE)
    model, base_model = build_model()
    model.summary()

    train_model(model, base_model, train_ds, val_ds)

    logger.info("Tuning classification threshold on validation set")
    best_threshold = find_best_threshold(model, val_ds, val_df)
    logger.info("Best threshold: %.2f", best_threshold)

    logger.info("Evaluating on test set")
    test_probs = model.predict(test_ds).ravel()
    test_preds = (test_probs >= best_threshold).astype(int)
    test_true  = test_df["label"].values

    acc = accuracy_score(test_true, test_preds)
    wf1 = f1_score(test_true, test_preds, average="weighted")

    print("\n=== Test Set Results ===")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Weighted F1:  {wf1:.4f}")
    print(f"Threshold:    {best_threshold:.2f}")
    print()
    print(classification_report(test_true, test_preds, target_names=["no_pothole", "pothole"]))

    metrics = {
        "accuracy": acc,
        "weighted_f1": wf1,
        "best_threshold": best_threshold,
    }

    logger.info("Saving artifacts to %s", SAVED_MODEL_DIR)
    model.save(SAVED_MODEL_DIR / "model.keras")
    joblib.dump(best_threshold, SAVED_MODEL_DIR / "threshold.joblib")
    joblib.dump(metrics,        SAVED_MODEL_DIR / "metrics.joblib")
    logger.info("Done.")


if __name__ == "__main__":
    main()
