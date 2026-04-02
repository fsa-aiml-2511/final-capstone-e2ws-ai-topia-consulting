"""
Shared Data Pipeline
====================
Shared data loading and preprocessing functions used across all models.
Put your common data cleaning, feature engineering, and splitting logic here.

Usage from any model:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipelines.data_pipeline import load_raw_data, preprocess, split_data
"""
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import re

from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

#Create interactive map
import os
import folium 
from folium.plugins import HeatMapWithTime
from folium.plugins import FastMarkerCluster

# Zipcode lookup
try:
    from uszipcode import SearchEngine
    ZIPCODE_SEARCH_AVAILABLE = True
except (ImportError, AttributeError) as e:
    ZIPCODE_SEARCH_AVAILABLE = False

#Set Regions
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Environmental data
import openmeteo_requests
from astral import Observer
from astral.sun import sun

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# =============================================================================
# HINT 1: Loading the Accident Data
# =============================================================================
def load_raw_data(filename):
    """Load a raw CSV file from data/raw/.

    Args:
        filename: Name of the CSV file (e.g., "patient_encounters_2023.csv")

    Returns:
        pandas DataFrame

    Example:
        df = load_raw_data("patient_encounters_2023.csv")
    """
    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Make sure you've downloaded the data to data/raw/"
        )
    
    return pd.read_csv(filepath)

# ============================================================================================================================
# Common Data Cleaning & Feature Engineering
# ============================================================================================================================
def clean_data(df):
    """Apply common data cleaning steps.

    Things to handle:
    - Missing value encoding (e.g., '?' -> NaN)
    - Data type conversions
    - Remove duplicates
    - Drop irrelevant columns

    Returns:
        Cleaned DataFrame
    """
    #drop Duplicates
    df = df.drop_duplicates()

    # cleaning - change all text to lower case for consistency
    df = df.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)

    return df

# =============================================================================
# HINT 2: Temporal Feature Engineering
# =============================================================================
def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Time patterns are among the strongest predictors of accident severity.

    Features to extract:
    - Hour of day (rush hour vs. off-peak)
    - Day of week (weekday vs. weekend)
    - Month (seasonal patterns — winter ice, summer heat)
    - Duration of traffic impact
    - Is it dark? (Sunrise_Sunset column helps, but you can derive from time too)
    """
    df['hour'] = df['Start_Time'].dt.hour
    df['day_of_week'] = df['Start_Time'].dt.dayofweek
    df['month'] = df['Start_Time'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Rush hour flags
    df['is_morning_rush'] = df['hour'].between(7, 9).astype(int)
    df['is_evening_rush'] = df['hour'].between(16, 19).astype(int)
    df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)

    # Duration of traffic impact (in minutes)
    if 'End_Time' in df.columns:
        df['duration_min'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
        # Cap extreme values
        df['duration_min'] = df['duration_min'].clip(0, 1440)  # Max 24 hours

    return df

# =============================================================================
# Split data into train and validation sets
# =============================================================================
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets with stratification.
    Verifies the balance of the split automatically.
    """
    # 1. Perform the split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    # 2. Verify split size (Your requested check)
    print("--- Data Split Component ---")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # 3. Verify stratification/class distribution
    print(f"\nTraining class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        percentage = (c / len(y_train)) * 100
        print(f" Class {u}: {c} samples ({percentage:.1f}%)")
    
    return X_train, X_test, y_train, y_test

# =============================================================================
# Save processed data 
# =============================================================================
def save_processed_data(df, filename):
    """Save processed data to data/processed/.

    Args:
        df: Processed DataFrame
        filename: Output filename (e.g., "encounters_processed.csv")
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / filename

    # Drop the file if it already exists
    if output_path.exists():
        output_path.unlink()
        print(f"Existing file {filename} dropped.")

    # Save the new version
    df.to_csv(output_path, index=False)
    print(f"Saved fresh processed data to {output_path}")

# =============================================================================
# Load previously processed data 
# =============================================================================
def load_processed_data(filename):
    """Load previously processed data from data/processed/.

    Args:
        filename: Name of the processed CSV file

    Returns:
        pandas DataFrame
    """
    filepath = PROCESSED_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data not found: {filepath}\n"
            f"Run the data pipeline first to generate processed data."
        )
    return pd.read_csv(filepath)

# =============================================================================
# Changes true and false to 1 and 0
# =============================================================================
def convert_bools_to_ints(df):
    # 1. Find all columns that are of type 'bool'
    bool_cols = df.select_dtypes(include=['bool']).columns
    
    # 2. Convert only those columns to integer (True -> 1, False -> 0)
    df[bool_cols] = df[bool_cols].astype(int)
    
    return df

# ==================================================================================================
# Drop columns where a single value occupies more than 99% of the rows
# ==================================================================================================
def drop_low_variance_columns(df, threshold=0.99):
    """
    This function removes features that are functionally constant 
    (i.e., where a single value occupies more than the threshold percentage of the total rows),
    because a column where almost every row is the same provides no "contrast" for the model to learn from.
    """
    # Identify columns to drop
    cols_to_drop = []
    
    for col in df.columns:
        # Get the proportion of the most frequent value
        top_value_ratio = df[col].value_counts(normalize=True).iloc[0]
        
        if top_value_ratio > threshold:
            cols_to_drop.append(col)
            
    print(f"Dropped {len(cols_to_drop)} columns with > {threshold*100}% dominance.")
    print(f"Columns dropped: {cols_to_drop}")
    
    return df.drop(columns=cols_to_drop)

# ==================================================================================================
# gets the process data and prints out the range and std of the target variable for interpretation
# ==================================================================================================
def get_data_and_process_target(file_path, target_column):
    """
    Loads and inspects the processed dataset.
    """
    try:
        # Load the CSV
        load_processed_data(file_path)
        df = load_processed_data(file_path)
        
        # Basic inspection
        print(f"--- Data Successfully Loaded ---")
        print(f"Data shape: {df.shape}")
        
        # Calculate target stats for interpretation
        if target_column in df.columns:
            target_range = df[target_column].max() - df[target_column].min()
            target_std = df[target_column].std()
            
            print(f"Target Column: '{target_column}'")
            print(f"Target range: {target_range:,.2f}")
            print(f"Target std: {target_std:,.2f}")
            
            # Return the dataframe and the stats as a dictionary
            stats = {
                'range': target_range,
                'std': target_std
            }
            return df, stats
        else:
            print(f"Error: Target column '{target_column}' not found in dataframe.")
            return df, None
            
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None, None

# ==================================================================================================
# Feature Scaling   
# ==================================================================================================
def scale_features(X_train, X_test):
    """
    Standardizes features by removing the mean and scaling to unit variance.
    Returns DataFrames to preserve column names and indices.
    """
    # 1. Initialize the Scaler
    scaler = StandardScaler()
    
    # 2. Fit on TRAIN and Transform BOTH
    # (We only 'fit' on train to prevent data leakage from the test set)
    X_train_scaled_array = scaler.fit_transform(X_train)
    X_test_scaled_array = scaler.transform(X_test)
    
    # 3. Convert back to DataFrames to keep columns and index
    X_train_scaled = pd.DataFrame(
        X_train_scaled_array, 
        columns=X_train.columns, 
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        X_test_scaled_array, 
        columns=X_test.columns, 
        index=X_test.index
    )
    
    # 4. Track metadata for the app
    SELECTED_FEATURES = X_train.columns.tolist()
    
    print("--- Scaling Component ---")
    print(f"Features scaled successfully!")
    print(f"Scaler fitted on {len(SELECTED_FEATURES)} features.")
    
    return X_train_scaled, X_test_scaled, scaler, SELECTED_FEATURES

# ==================================================================================================
# Label Encoding for Classification Targets
# ==================================================================================================
def label_encode_target(y):
    """
    Standardizes target labels to start at 0.
    Prints the mapping for verification.
    """
    # 1. Initialize the Encoder
    label_encoder = LabelEncoder()
    
    # 2. Fit and Transform the target
    y_encoded = label_encoder.fit_transform(y)
    
    # 3. Verify and Print encoding (Your requested check)
    print("--- Label Encoding Component ---")
    print("Label mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {label} -> {i}")
    
    # Return the encoded array and the encoder object for inverse mapping later
    return y_encoded, label_encoder