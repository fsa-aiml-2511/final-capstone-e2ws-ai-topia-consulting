"""
Shared Data Pipeline
====================
Shared data loading and preprocessing functions used across all models.
"""
import sys
import pandas as pd
from pathlib import Path
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ==========================================================================================
# 1. MASTER PIPELINE FUNCTION
# ==========================================================================================

def get_cleaned_img_data(data_path_suffix, img_size=(28, 28), num_classes=2, test_size=0.2):
    """
    ONE-STEP PIPELINE:
    1. Loads raw images from RAW_DATA_DIR.
    2. Calls preprocess_image_data (Normalization, Reshaping, Encoding).
    3. Splits data into Train and Test sets.
    4. Returns 4 variables: X_train, X_test, y_train, y_test.
    """
    class_map = {"negative": 0, "positive": 1}
    images = []
    labels = []
    
    filepath = RAW_DATA_DIR / data_path_suffix
    if not filepath.exists():
        print(f"❌ Error: Path {filepath} not found.")
        return None, None, None, None

    # --- LOAD RAW DATA ---
    print(f"Loading raw images from {filepath}...")
    for folder in os.listdir(filepath):
        folder_path = os.path.join(filepath, folder)
        if os.path.isdir(folder_path) and folder.lower() in class_map:
            label = class_map[folder.lower()]
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    images.append(img)
                    labels.append(label)

    X_raw = np.array(images)
    y_raw = np.array(labels)

    if len(X_raw) == 0:
        print("❌ No images loaded.")
        return None, None, None, None

    # --- PREPROCESS (MATH HAPPENS ONCE HERE) ---
    X_all, y_all = preprocess_image_data(X_raw, y_raw, img_size, num_classes)

    # --- DYNAMIC SPLIT ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, 
        test_size=test_size, 
        random_state=42, 
        stratify=y_all
    )

    # --- SAVE ---
    save_prepped_images(X_train, y_train, prefix="train")
    save_prepped_images(X_test, y_test, prefix="test")
    
    print(f"✅ Success! Split into {len(X_train)} train and {len(X_test)} test images.")
    return X_train, X_test, y_train, y_test

# ==========================================================================================
# 2. PREPROCESSING (The 'Chef's Prep')
# ==========================================================================================

def preprocess_image_data(X, y, img_size=(28, 28), num_classes=2):
    """
    Prepares images for the CNN. Includes a 'Safety Gate' to prevent 
    double-normalization and double-encoding.
    """
    print("Starting Preprocessing...")

    # 1. Normalize pixel values (Safety Gate: Only divide if Max > 1)
    if X.max() > 1.0:
        X_prepped = X.astype('float32') / 255.0
        print(f"  - Normalized: Pixel range is now {X_prepped.min()} to {X_prepped.max()}")
    else:
        X_prepped = X
        print(f"  - Skip Normalization: Data already in range {X_prepped.min()} to {X_prepped.max()}")

    # 2. Reshape (Check if channel dimension 1 is already there)
    if len(X_prepped.shape) == 3:
        X_prepped = X_prepped.reshape(-1, img_size[0], img_size[1], 1)
        print(f"  - Reshaped: New dimensions are {X_prepped.shape}")
    else:
        print(f"  - Skip Reshape: Dimensions already {X_prepped.shape}")

    # 3. One-hot encode (Safety Gate: Only encode if y is 1D)
    if len(y.shape) == 1:
        y_prepped = to_categorical(y, num_classes)
        print(f"  - One-hot encoded: Label shape is now {y_prepped.shape}")
    else:
        y_prepped = y
        print(f"  - Skip Encoding: Label shape already {y_prepped.shape}")

    return X_prepped, y_prepped

# ==========================================================================================
# 3. UTILITIES: SAVE, VISUALIZE, & AUGMENT
# ==========================================================================================

def save_prepped_images(X, y, prefix="train"):
    """ Drops existing files and saves fresh NumPy arrays. """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    x_path = PROCESSED_DATA_DIR / f"X_{prefix}.npy"
    y_path = PROCESSED_DATA_DIR / f"y_{prefix}.npy"
    
    for path in [x_path, y_path]:
        if path.exists():
            path.unlink()
            print(f"🗑️  Dropped existing file: {path.name}")
    
    np.save(x_path, X)
    np.save(y_path, y)
    print(f"✅ Saved to: {PROCESSED_DATA_DIR}")

def get_pothole_augmenter():
    """ Live training augmentation. Use during model.fit() """
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

def visualize_samples(images, labels, title="Sample Pothole Images"):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i].reshape(images[i].shape[0], images[i].shape[1]), cmap='gray')
            # Handle one-hot or raw labels for the title
            lbl = np.argmax(labels[i]) if len(labels[i].shape) > 0 else labels[i]
            ax.set_title(f"Label: {lbl}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_class_distribution(y_data, title="Class Distribution"):
    y_counts = np.argmax(y_data, axis=1) if len(y_data.shape) > 1 else y_data
    unique, counts = np.unique(y_counts, return_counts=True)
    plt.figure(figsize=(10, 5))
    plt.bar(unique, counts, color='#8B7355', edgecolor='#5D4E37')
    plt.xticks(unique)
    plt.title(title)
    plt.show()
    for label, count in zip(unique, counts):
        name = "Pothole" if label == 1 else "Normal"
        print(f"  {name}: {count:,} images")