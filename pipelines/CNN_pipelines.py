import sys
from pathlib import Path
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

print(f"TensorFlow version: {tf.__version__}")
print("Setup complete!")

# Settings
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

root_path = Path.cwd().parent 
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

# ========================================================================================
# Builds and Compiles a Keras Sequential (CNN) model.
# ========================================================================================
def build_pothole_cnn(input_shape=(28, 28, 1), num_classes=2):
    """
    Step 5: Building Our CNN
    Creates a Sequential model with Feature Learning and Classification blocks.
    Input → [Conv + ReLU → Pooling] × 2 → Flatten → Dense → Output
    """
    model = keras.Sequential([
        
        # === FEATURE LEARNING BLOCK 1 ===
        # Convolutional Layer: 32 filters, each 3x3
        # These filters will learn to detect basic features like edges and curves
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        
        # Max Pooling: Reduce size by half (2x2 window)
        # Keeps the strongest activations, reduces computation
        layers.MaxPooling2D((2, 2)),
        
        # === FEATURE LEARNING BLOCK 2 ===
        # More filters (64) to detect more complex patterns
        # Deeper layers learn higher-level features (shapes, shadows of potholes)
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Max Pooling again: Further reduce spatial dimensions
        layers.MaxPooling2D((2, 2)),
        
        # === CLASSIFICATION BLOCK ===
        # Flatten: Convert 2D feature maps to 1D vector
        # This is the bridge between feature learning and classification
        layers.Flatten(),
        
        # Dense Layer: 128 neurons to learn patterns from extracted features
        layers.Dense(128, activation='relu'),
        
        # Dropout: Randomly turn off 50% of neurons during training
        # This prevents overfitting (remember: memorizing vs understanding!)
        layers.Dropout(0.5),
        
        # Output Layer: 2 neurons (Normal vs Pothole), softmax gives probabilities
        layers.Dense(num_classes, activation='softmax')
    ])

    # Step 6: Compile the Model
    # Optimizer: Adam (efficient gradient descent)
    # Loss: Categorical Crossentropy (standard for classification)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("✅ Model built and compiled successfully!")
    return model

# ==========================================================================================
# The Training & Evaluation Wrapper Keras Sequential (CNN) model.
# ==========================================================================================
def train_and_evaluate_pothole_model(model, X_train, y_train, X_test, y_test, epochs=10):
    """
    Combines Step 7 (Training) and Step 8 (Evaluation) into one step.
    """
    # --- Step 7: Train the Model ---
    print(f"🚀 Starting training for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=128,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # --- Step 8: Evaluate on Test Data ---
    print("\n🧐 Evaluating on unseen test data...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    # --- Print Results ---
    print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"\nOur CNN correctly classifies {test_accuracy*100:.2f}% of images it has never seen before!")

    return history

# ==========================================================================================
# Visualize Training Progress
# ==========================================================================================
def plot_training_results(history):
    """
    Step 7 Visualization: Plots Accuracy and Loss side-by-side.
    Helps identify if the model is learning or overfitting.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Accuracy plot ---
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='#8B7355')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#C65D00')
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # --- Loss plot ---
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2, color='#8B7355')
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#C65D00')
    axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()