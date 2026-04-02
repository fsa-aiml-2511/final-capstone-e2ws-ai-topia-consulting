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
import numpy as np
from collections import Counter
import re

from pipelines.data_pipeline import create_temporal_features
# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ===================================================================================================================================
# Processing & Feature Engineering for 311 Service Request Data
# ===================================================================================================================================
def complaints_engineer_features(df):
    """Create new features from existing columns.

    Examples:
    - Parse datetime columns -> hour, day_of_week, month
    - Create binary flags from categorical data
    - Bin continuous variables into categories
    - Interaction features

    Returns:
        DataFrame with new feature columns
    """
    #Create temporal features (e.g., hour of day, day of week, etc.)
    df = create_temporal_features(df) 
    return df



