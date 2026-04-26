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
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder

# Visualization
import matplotlib.pyplot as plt
from seaborn.objects import Plot
from sklearn.inspection import permutation_importance

# Sklearn - evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score, classification_report, confusion_matrix,  RocCurveDisplay, PrecisionRecallDisplay
)
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
    if 'Start_Time' in df.columns:
        df['hour'] = df['Start_Time'].dt.hour
        df['day_of_week'] = df['Start_Time'].dt.dayofweek
        df['month'] = df['Start_Time'].dt.month
    
    if 'day_of_week' in df.columns:
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Rush hour flags
    if 'hour' in df.columns:
        df['is_morning_rush'] = df['hour'].between(7, 9).astype(int)
        df['is_evening_rush'] = df['hour'].between(16, 19).astype(int)

    if 'is_morning_rush' in df.columns and 'is_evening_rush' in df.columns:
        df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)

    # Duration of traffic impact (in minutes)
    if 'End_Time' in df.columns:
        df['duration_min'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
        # Cap extreme values
        df['duration_min'] = df['duration_min'].clip(0, 1440)  # Max 24 hours

    # Handle Created Date
    if 'created_date' in df.columns:
        # Convert to datetime first to avoid the AttributeError
        df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
        
        # Use specific names so they don't get overwritten
        df['created_hour'] = df['created_date'].dt.hour
        df['created_day_of_week'] = df['created_date'].dt.dayofweek
        df['created_month'] = df['created_date'].dt.month

    # Handle Closed Date
    if 'closed_date' in df.columns:
        # Convert to datetime first
        df['closed_date'] = pd.to_datetime(df['closed_date'], errors='coerce')
        
        # Use 'closed_' prefix
        df['closed_hour'] = df['closed_date'].dt.hour
        df['closed_day_of_week'] = df['closed_date'].dt.dayofweek
        df['closed_month'] = df['closed_date'].dt.month
        
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

# =============================================================================
# This helper function prints detailed diagnostic reports for the model
# =============================================================================
def print_model_report(y_test, y_test_pred, model_name):
    """Prints the official Scikit-Learn Classification 
    Example Output:
                  precision    recall  f1-score   support

               1       0.73      0.15      0.25       870
               2       0.85      0.96      0.91     79534
               3       0.70      0.39      0.50     16808
               4       0.62      0.11      0.18      2646

        accuracy                           0.84     99858
       macro avg       0.72      0.40      0.46     99858
    weighted avg       0.82      0.84      0.81     99858
    """
    print(f"\n" + "="*60)
    print(f" MODEL: {model_name}")
    print("="*60)
    
    # This produces the exact precision, recall, f1-score, support table
    # We don't hardcode target_names so it stays dynamic for different datasets
    report = classification_report(y_test, y_test_pred)
    print(report)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("="*60 + "\n")

# =============================================================================
# This helper function extracts, prints, and plots feature importances
# =============================================================================
def plot_feature_importance(trained_model, X_val, y_val, model_name, top_n=30):
    """
    Calculates Permutation Importance for any model type.
    Works for HistGradientBoosting, SVM, KNN, and Tree-based models.
    """
    print(f"\nCalculating feature importance for {model_name}...")
    
    # Calculate Permutation Importance
    # We use the validation/test set here to see which features actually matter for prediction
    result = permutation_importance(
       trained_model, X_val, y_val, n_repeats=5, random_state=42, n_jobs=-1
    )

    # Map to column names
    feat_imp = pd.DataFrame({
        "feature": X_val.columns if isinstance(X_val, pd.DataFrame) else [f"Feature {i}" for i in range(X_val.shape[1])],
        "importance": result.importances_mean
    })

    # Sort and filter top N
    feat_imp = feat_imp.sort_values("importance", ascending=False).head(top_n)

    # Print results
    print(f"Top {top_n} features for {model_name}:")
    for i, row in feat_imp.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(feat_imp["feature"][::-1], feat_imp["importance"][::-1], color='teal')
    plt.xlabel("Decrease in Accuracy (Permutation Importance)")
    plt.ylabel("Feature")
    plt.title(f"{model_name} - Top {top_n} Feature Importances")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return feat_imp

# =============================================================================
# This helper function plots how confident the model is in its predictions
# =============================================================================
def plot_prediction_probabilities(trained_model, X_test, model_name):
    """Plots a histogram of the max predicted probabilities for each sample."""
    # Check if model supports probability estimates
    if not hasattr(trained_model, "predict_proba"):
        print(f"{model_name} does not support predict_proba.")
        return

    # Get the probabilities for each class
    probs = trained_model.predict_proba(X_test)
    
    # Take the highest probability for each prediction (the confidence level)
    max_probs = np.max(probs, axis=1)

    plt.figure(figsize=(10, 6))
    plt.hist(max_probs, bins=30, color='coral', edgecolor='black', alpha=0.7)
    plt.axvline(x=0.5, color='red', linestyle='--', label='Random Guessing (50%)')
    plt.xlabel('Predicted Probability (Confidence)')
    plt.ylabel('Number of Samples')
    plt.title(f'Prediction Confidence Distribution - {model_name}')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()



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
# Add project root to sys.path
root_path = Path.cwd().parent 
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

#Import
from pipelines.data_pipeline import convert_bools_to_ints, create_temporal_features
# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ============================================================================================================================
# Processing & Feature Engineering for City Traffic Accident Data
# ============================================================================================================================
def accident_engineer_features(df):
    """
    Production-ready feature engineering pipeline.
    
    USAGE:
        df = accident_engineer_features(df)
    """
    # 1. Basic Cleaning & Dropping
    df = df.drop(columns=['Country', 'ID', 'Source'], errors='ignore')

    #Cleans and fills empty columns
    df = accident_engineer_empty_columns(df)

    #Feature Engineering non-numertical columns
    df= description_word_count(df)      #Engineer a feature that counts the number of words in the Description column, which may correlate with accident severity or complexity.

    #Categorize Weather Conditions
    df = process_weather_features(df)

    #one-hot encode 
    if 'Region' in df.columns:
        df = pd.concat([df.drop(columns=['Region']), pd.get_dummies(df['Region'], prefix='region', dummy_na=False, dtype=int)], axis=1)
    if 'Wind_Direction' in df.columns:
        df = pd.concat([df.drop(columns=['Wind_Direction']), pd.get_dummies(df['Wind_Direction'], prefix='wind', dummy_na=False, dtype=int)], axis=1)

    df = process_road_features(df)              #Creating aggregate features for road and traffic

    # Find top 5 zipcde in each region and group the rest into "other" category
    df = create_zipcode_features(df)

    # Group cities outside of top 20 into "Other" category to reduce cardinality
    df= encode_top_geo_features(df)

    # Drop any remaining irrelevant or redundant columns (e.g., Street, if it was too noisy and we filled geographic details from lat/lng)
    df = df.drop(columns=['State', 'Zipcode', 'City', 'County', 'Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng'], errors='ignore')

    #logic for the matching dangerous weather patterns
    df = dangerous_conditions_score(df)

    #Engineer aggregate features for road conditions (e.g., total_road_features, has_traffic_control)
    df= engineer_road_features(df) 

    df=convert_bools_to_ints(df) #Convert boolean columns to integers for modeling
    #Retrun the processed DataFrame
    return df

# =============================================================================
# Cleans and fills empty columns for City Traffic Accident Data
# =============================================================================
def accident_engineer_empty_columns(df):
    """
    Cleans and fills empty columns using a combination of logic, 
    lookup tables, 
    and local calculations.
    
    USAGE:
        df = accident_engineer_empty_columns(df)
    """

    #Datetime Conversion
    time_cols = ['Start_Time', 'End_Time', 'Weather_Timestamp']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    #Time Imputation
    df['Start_Time'] = df['Start_Time'].fillna(df['Weather_Timestamp'])             #If Start_Time is missing, use Weather_Timestamp as a proxy (assuming weather data is timestamped at the time of the accident)
    df['End_Time'] = df['End_Time'].fillna(df['Weather_Timestamp'])                 #If End_Time is missing, use Weather_Timestamp as a proxy (assuming weather data is timestamped at the time of the accident)
    df['Weather_Timestamp'] = df['Weather_Timestamp'].fillna(df['Start_Time'])      #If Weather_Timestamp is missing, use Start_Time as a proxy (assuming weather data is timestamped at the time of the accident)

    # Temporal Features (Hour, Month, Rush Hour)
    df = create_temporal_features(df)                                               #Extract hour of day, day of week, month, and rush hour flags from Start_Time to capture time-based patterns in accidents

    #Fill Coordinates & Geographic Details
    if 'End_Lat' in df.columns:
        df['End_Lat'] = df['End_Lat'].fillna(df['Start_Lat'])
    if 'End_Lng' in df.columns:
        df['End_Lng'] = df['End_Lng'].fillna(df['Start_Lng'])
    
     #Geographic & Regional Logic
    #df = fill_geographic_data(df)                                               #Use ZIP code lookup or reverse geocoding to fill missing geographic details (e.g., city, county) based on available lat/lng or zip code data
    df = df.drop(columns=['Street'], errors='ignore')                           #Drop Street column after filling geographic details, as it may be too noisy or sparse to be useful
    df = add_census_regions(df)                                                     #Add Census Region based on State to capture regional patterns in accidents
    df = create_cluster_regions(df, n_clusters=10)                                  #Create geographic clusters (e.g., using KMeans on lat/lng) to capture local accident hotspots 
    df = add_intra_region_distances(df, cluster_col='Geo_Cluster')                  #Calculate distance to cluster centers to capture how far an accident is from local hotspots
    #WEATHER & ENVIRONMENT DATA - Fast processing
    df = fast_environmental_data(df)                                            #Use local calculations (e.g., sunrise/sunset times based on lat/lng and date) to fill missing environmental data
    
    # Drop unnamed numeric-indexed columns
    df = df.loc[:, ~df.columns.astype(str).str.match(r'^\d+$')]                 # Drop unnamed numeric-indexed columns
    

    return df

# =============================================================================
# Generate a heatmap 
# =============================================================================
def generate_hourly_heatmap(data, filename=None):
    """Generate a heatmap to visualize the density of accidents over time and location

    Args:
        City Traffic Accidents

    Saves map to display in app:
        pandas DataFrame
    """
    if filename is None:
        filename = str(PROJECT_ROOT / "data" / "maps" / "interactive_traffic_map.html")
    
    #Automatically create the directory if it doesn't exist
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    #Remove existing file if it exists to ensure overwrite
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Removed existing file: {filename}")

    #Prepare the hourly data
    hourly_data = []
    for hour in range(24):
        subset = data[data['hour'] == hour]
        points = subset[['Start_Lat', 'Start_Lng']].dropna().values.tolist()
        hourly_data.append(points)
    
    #Create the map
    m = folium.Map(location=[37.0902, -95.7129], zoom_start=4)
    HeatMapWithTime(
        hourly_data, 
        index=[f"{h}:00" for h in range(24)],
        radius=10, 
        auto_play=True, 
        max_opacity=0.8
    ).add_to(m)
    
    #Save
    m.save(filename)
    print(f"Map successfully saved to: {filename} (overwrote existing file)")
    return m

# =============================================================================
# Generate a severity map 
# =============================================================================
def generate_accident_map(data, filename=None):

    """Generate a map to visualize the locations of accidents and their severity

    Args:
        City Traffic Accidents

    Saves map to display in app:
        pandas DataFrame
    """
    if filename is None:
        filename = str(PROJECT_ROOT / "data" / "maps" / "accident_map.html")
    
    #Automatically create the directory if it doesn't exist
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    #Remove existing file if it exists to ensure overwrite
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Removed existing file: {filename}")

    #Prepare the data data
    df_sample = data[['Start_Lat', 'Start_Lng']].dropna()
    
    #Create the map United State Only
    m = folium.Map(location=[37.0902, -95.7129], zoom_start=4)
    FastMarkerCluster(data=df_sample.values.tolist()).add_to(m)
    
    #Save
    m.save(filename)
    print(f"Map successfully saved to: {filename} (overwrote existing file)")
    return m

# =============================================================================
# Calculate wind chill based on rules
# =============================================================================
def calculate_wind_chill(temp, speed, chill):
    """Calculate wind chill based on rules:
    - If wind_chill is not null, keep current value
    - If wind_chill is null and temp > 50°F, use temperature value
    - If wind_chill is null and temp <= 50°F, calculate using formula if wind_speed > 3 mph
    
    Formula: WindChill = 35.74 + 0.6215T - 35.75(V^0.16) + 0.4275T(V^0.16)
    T = Air Temperature (°F)
    V = Wind Speed (mph)
    
    Args:
        temp: Temperature(F) - scalar or Series
        speed: Wind_Speed(mph) - scalar or Series
        chill: Wind_Chill(F) - scalar or Series
    
    Returns:
        Calculated wind chill value(s)
    """
    # For each row, apply the logic
    if pd.isna(chill):
        # If wind_chill is null
        if temp > 50:
            # If temp above 50°F, use temperature as wind chill
            return temp
        elif speed > 3:
            # If temp <= 50°F and wind speed > 3 mph, calculate formula
            return 35.74 + (0.6215 * temp) - (35.75 * (speed**0.16)) + (0.4275 * temp * (speed**0.16))
        else:
            # If temp <= 50°F and wind speed <= 3 mph, use temperature
            return temp
    else:
        # Keep existing wind chill value if not null
        return chill

# =============================================================================
# Fill missing Airport_Code and Zipcode 
# =============================================================================
def airport_code_to_zip(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing Airport_Code and Zipcode values using bidirectional lookup.
    
    - For rows where Airport_Code is null, uses Zipcode to lookup the nearest airport code
    - For rows where Zipcode is null, uses Airport_Code to lookup the nearest zip code
    
    Uses the zip_to_airport_lookup.csv file for mapping.

    Args:
        df: pandas DataFrame with 'Airport_Code' and 'Zipcode' columns

    Returns:
        DataFrame with filled Airport_Code and Zipcode values
    """
    try:
        lookup_file_path = PROJECT_ROOT / "data" / "lookup_tables" / "airport_lookup.csv"
        if not lookup_file_path.exists():
            print(f"Lookup file not found: {lookup_file_path}. Skipping airport/zip code fill.")
            return df
        
        lookup = pd.read_csv(lookup_file_path, dtype=str)
        
        # Create both forward and reverse mappings
        zip_to_airport = dict(zip(lookup["zip_code"], lookup["nearest_airport_iata"]))
        
        # Fill missing Airport_Code using Zipcode lookup
        ac_mask = df['Airport_Code'].isna()
        filled_ac = ac_mask.sum()
        df.loc[ac_mask, 'Airport_Code'] = df.loc[ac_mask, 'Zipcode'].map(zip_to_airport)
        
    except Exception as e:
        print(f"Error filling Airport_Code/Zipcode: {e}. Continuing without this step.")
    
    return df

# =============================================================================
# Fill in Street/City/Zip/Timezone
# =============================================================================
def fill_geographic_data(df: pd.DataFrame) -> pd.DataFrame:
    search = SearchEngine()
    
    # Identify rows where Street is missing (along with City/Zip/Timezone)
    mask = (df['Street'].isna()) | (df['Street'].astype(str).str.lower().isin(['none', 'nan', 'missing'])) | \
           (df['City'].isna()) | (df['Timezone'].isna())

    def repair_geo_details(row):
        result = search.by_coordinates(row['Start_Lat'], row['Start_Lng'], radius=50, returns=1)
        if result:
            res = result[0]
            # Use City Center (Major City) as a placeholder for Street if Street is missing
            return pd.Series({
                'Zipcode': res.zipcode,
                'City': res.major_city.lower(),
                'County': res.county.lower(),
                'Timezone': f"us/{res.timezone.lower()}" if res.timezone else row['Timezone'],
            })
        return pd.Series({'Zipcode': '00000', 'City': 'unknown', 'County': 'unknown', 'Timezone': 'missing'})

    # Apply the logic
    df.loc[mask, ['Zipcode', 'City', 'County', 'Timezone']] = df[mask].apply(repair_geo_details, axis=1)
    
    return df

# =============================================================================
# Create an inverse mapping for fast lookup
# =============================================================================
def add_census_regions(df):
    regions = {
        'Northeast': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA'],
        'Midwest': ['IL', 'IN', 'IA', 'KS', 'MI', 'MN', 'MS', 'NE', 'ND', 'OH', 'SD', 'WI'],
        'South': ['AL', 'AR', 'DE', 'FL', 'GA', 'KY', 'LA', 'MD', 'MS', 'NC', 'OK', 'SC', 'TN', 'TX', 'VA', 'WV'],
        'West': ['AK', 'AZ', 'CA', 'CO', 'HI', 'ID', 'MT', 'NV', 'NM', 'OR', 'UT', 'WA', 'WY']
    }
    # Create an inverse mapping for fast lookup
    state_to_region = {state: region for region, states in regions.items() for state in states}
    
    df['Region'] = df['State'].str.upper().map(state_to_region).fillna('Other')
    return df

# =============================================================================
# Finds 10 high-density regions
# =============================================================================
def create_cluster_regions(df, n_clusters=10):
    """
    Create geographic clusters from Start_Lat / Start_Lng.

    Robust to:
    - very small datasets
    - missing coordinates
    - prediction-time batches with fewer rows than requested clusters
    """
    df = df.copy()

    required_cols = ['Start_Lat', 'Start_Lng']
    if not all(col in df.columns for col in required_cols):
        df['Geo_Cluster'] = 0
        return df

    coords = df[required_cols].dropna()

    # If no usable coordinates, assign a default cluster
    if coords.empty:
        df['Geo_Cluster'] = 0
        return df

    # KMeans cannot have more clusters than samples
    effective_clusters = min(n_clusters, len(coords))

    # If only 1 valid row, just assign cluster 0
    if effective_clusters <= 1:
        df['Geo_Cluster'] = 0
        return df

    kmeans = KMeans(n_clusters=effective_clusters, random_state=42)
    labels = kmeans.fit_predict(coords)

    # Default all rows to 0, then fill valid-coordinate rows with labels
    df['Geo_Cluster'] = 0
    df.loc[coords.index, 'Geo_Cluster'] = labels

    return df

# =============================================================================
# Calculate the 'Hotspot' for each region
# =============================================================================
def add_intra_region_distances(df, cluster_col='Geo_Cluster'):
    """
    Calculates the distance from each accident to the center 
    of its assigned cluster/region.
    """
    #Calculate the 'Hotspot' (Centroid) for each region
    centroids = df.groupby(cluster_col)[['Start_Lat', 'Start_Lng']].mean().reset_index()
    centroids.columns = [cluster_col, 'Centroid_Lat', 'Centroid_Lng']
    
    #Merge these centers back into the main dataframe
    df = df.merge(centroids, on=cluster_col, how='left')
    
    # Calculate distance (Haversine or simple Euclidean approximation)
    # A degree is roughly 69 miles
    lat_diff = (df['Start_Lat'] - df['Centroid_Lat']) ** 2
    lng_diff = (df['Start_Lng'] - df['Centroid_Lng']) ** 2
    
    df['dist_from_reg_hotspot'] = np.sqrt(lat_diff + lng_diff) * 69
    
    # Drop the temporary centroid columns to keep the DF clean
    df.drop(columns=['Centroid_Lat', 'Centroid_Lng'], inplace=True)
    
    return df

# =============================================================================
# FAST environmental data: Uses local Astral + regional median filling
# =============================================================================
def fast_environmental_data(df):
    """
    FAST environmental data: Uses local Astral + regional median filling.
    Fills ALL weather with cluster-based medians (instant, deterministic).
    """
    
    # ===== SUN DATA (Local Astral calculations) =====
    mask = df['Sunrise_Sunset'].isna()
    if mask.any():
        print(f"  Calculating sun data for {mask.sum()} rows...")
        
        missing_data = df[mask].copy()
        missing_data['date'] = missing_data['Start_Time'].dt.date
        unique_days = missing_data[['date', 'Start_Lat', 'Start_Lng']].drop_duplicates()

        results_map = {}
        for _, row in unique_days.iterrows():
            try:
                obs = Observer(latitude=row['Start_Lat'], longitude=row['Start_Lng'])
                s = sun(obs, date=row['date'])
                results_map[(row['date'], row['Start_Lat'], row['Start_Lng'])] = s
            except:
                continue

        def apply_sun_logic(row):
            key = (row['Start_Time'].date(), row['Start_Lat'], row['Start_Lng'])
            if key not in results_map:
                return pd.Series({
                    'Sunrise_Sunset': np.nan,
                    'Civil_Twilight': np.nan,
                    'Nautical_Twilight': np.nan,
                    'Astronomical_Twilight': np.nan
                })
            
            s = results_map[key]
            acc_t = row['Start_Time'].tz_localize('UTC') if row['Start_Time'].tzinfo is None else row['Start_Time']
            
            return pd.Series({
                'Sunrise_Sunset': 'Day' if s['sunrise'] < acc_t < s['sunset'] else 'Night',
                'Civil_Twilight': 'Day' if s['dawn'] < acc_t < s['dusk'] else 'Night',
                'Nautical_Twilight': 'Day' if s['dawn'] < acc_t < s['dusk'] else 'Night',
                'Astronomical_Twilight': 'Day' if s['sunrise'] < acc_t < s['sunset'] else 'Night'
            })

        sun_updates = df[mask].apply(apply_sun_logic, axis=1)
        df.loc[mask, sun_updates.columns] = sun_updates
    
    # ===== WEATHER DATA (Regional median filling) =====
    numeric_weather_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 
                            'Wind_Speed(mph)', 'Precipitation(in)']
    categorical_weather_cols = ['Wind_Direction', 'Weather_Condition']
    
    print(f"  Filling weather with regional medians...")
    
    # Numeric columns: use median
    for col in numeric_weather_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df.groupby('Geo_Cluster')[col].transform('median'))
            df[col] = df[col].fillna(df[col].median())
    
    # Categorical columns: use mode (most common value)
    for col in categorical_weather_cols:
        if col in df.columns:
            # Mode-fill by cluster
            def fill_mode(group):
                if len(group) > 0:
                    mode_val = group.mode()[0] if len(group.mode()) > 0 else 'Unknown'
                    return group.fillna(mode_val)
                return group
            
            df[col] = df.groupby('Geo_Cluster')[col].transform(fill_mode)
            # Global fallback
            if df[col].isna().any():
                global_mode = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col] = df[col].fillna(global_mode)
    
    return df

# =============================================================================
# Identify top 20 words in the Descriptor column and features for each
# =============================================================================
def description_word_count(df):
    """Add features for the top 20 words in the Description column."""
    if 'Description' in df.columns:
        # Combine all descriptions, lowercase them, and find all words
        all_text = ' '.join(df['Description'].dropna()).lower()
        words = re.findall(r'\w+', all_text)
        
        # Filter out stop words
        stop_words = {'on', 'at', 'the', 'and', 'of', 'to', 'in', 'from', 'near', 'i', 'rd', 'st', 'ave', 'blvd', 'hwy', 'highway', 'street', 'road', 'due', 'us', 'ca', 'la', 'ny', 'tx', 'fl', 'il', 'wa', 'pa', 'oh', 'mi', 'ga', 'nc', 'nj', 'va', 'ma', 'az', 'co', 'nv', 'with', 'dr', 's', 'n', 'e', 'w', '95'}
        keywords = Counter([w for w in words if w not in stop_words])
        
        # Get top 20 words
        top_20_words = [word for word, count in keywords.most_common(20)]
        
        # Create columns for each top word - count occurrences in each description
        for word in top_20_words:
            df[f'word_{word}'] = df['Description'].fillna('').apply(lambda x: x.lower().count(word))
        
        # Drop the Description column
        df = df.drop(columns=['Description'], errors='ignore')
    
    return df

# =============================================================================
# HINT 3: Weather Feature Processing
# =============================================================================
def process_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weather is a major factor in accident severity.

    Missing values in weather columns are NOT random — they often mean:
    - Weather station was offline
    - Data wasn't available at the time of the accident
    - The weather API didn't return data for that location

    Strategy: Create a "weather_data_available" flag, then impute or drop.

    Key weather features:
    - Temperature(F): Freezing conditions are dangerous
    - Visibility(mi): Low visibility = more severe accidents
    - Precipitation(in): Rain/snow increases severity
    - Weather_Condition: Categorical (Clear, Rain, Snow, Fog, etc.)
    """
    weather_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)',
                    'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']

    # Create Data Availability Flag
    # Returns 1 if ANY data for these columns, 0 if all are NaN
    df['weather_data_available'] = df[weather_cols].notna().any(axis=1).astype(int)

    # Temperature Logic (Freezing check)
    if 'Temperature(F)' in df.columns:
        # Binary flag for freezing (dangerous for road traction)
        df['is_freezing'] = (df['Temperature(F)'] <= 32).astype(int)

    # Visibility Logic (Low visibility check)
    if 'Visibility(mi)' in df.columns:
        # Binary flag for hazardous visibility (under 2 miles)
        df['low_visibility_severity'] = (df['Visibility(mi)'] < 2).astype(int)

    # Precipitation Logic (Rain/Snow check)
    if 'Precipitation(in)' in df.columns:
        # Flag if there is any measurable precipitation
        df['has_precipitation'] = (df['Precipitation(in)'] > 0).astype(int)

    # Categorize the String Conditions
    if 'Weather_Condition' in df.columns:
        df['weather_cluster'] = df['Weather_Condition'].apply(categorize_weather)

    # One-Hot Encode 'weather_cluster' and drop the string version immediately
    # Use prefix to keep columns organized (e.g., weather_cluster_rain)
    df = df.join(pd.get_dummies(df.pop('weather_cluster'), prefix='weather_cluster', dtype=int))

    df = df.drop(columns=['Weather_Condition'], errors='ignore')    #Drop the Weather_Condition column after processing

    return df

# =============================================================================
# Weather Feature Processing - Categorization Logic
# =============================================================================
def categorize_weather(condition) -> str:
    """Group detailed weather conditions into broader categories."""
    if pd.isna(condition):
        return 'unknown'

    condition = str(condition).lower()

    # Priority order: Storms and Snow/Ice are higher risk than Rain
    if any(w in condition for w in ['clear', 'fair']):
        return 'clear'
    elif any(w in condition for w in ['cloud', 'overcast']):
        return 'cloudy'
    elif any(w in condition for w in ['rain', 'drizzle', 'shower']):
        return 'rain'
    elif any(w in condition for w in ['snow', 'sleet', 'ice', 'wintry']):
        return 'snow_ice'
    elif any(w in condition for w in ['fog', 'mist', 'haze', 'smoke']):
        return 'low_visibility'
    elif any(w in condition for w in ['thunder', 'storm']):
        return 'storm'
    else:
        return 'other'

# =============================================================================
# HINT 4: Road Feature Processing
# =============================================================================
def process_road_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    The dataset has 13 boolean road feature columns.

    These are already binary (True/False) and very useful for ML models.
    Consider creating aggregate features:
    - total_road_features: count of road features at the accident location
    - has_traffic_control: any of traffic signal, stop, give way, etc.
    """
    road_features = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
                     'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
                     'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']

    existing = [f for f in road_features if f in df.columns]

    # Total road features present
    df['n_road_features'] = df[existing].sum(axis=1)

    # Traffic control present
    control_features = ['Traffic_Signal', 'Stop', 'Give_Way', 'Traffic_Calming']
    existing_control = [f for f in control_features if f in df.columns]
    df['has_traffic_control'] = df[existing_control].any(axis=1).astype(int)

    return df

# =============================================================================
# HINT 5: Handling Severity Class Imbalance
# =============================================================================
def analyze_severity_distribution(df: pd.DataFrame):
    """
    Severity distribution is heavily imbalanced:
    - Severity 1: ~1-2% (very rare)
    - Severity 2: ~80% (dominant — this is your biggest challenge)
    - Severity 3: ~12-15%
    - Severity 4: ~5-8%

    This is a MAJOR challenge. If you just predict class 2 for everything,
    you'll get ~80% accuracy but your model is COMPLETELY USELESS.
    Weighted F1 is the real evaluation metric, not accuracy.

    Strategies:
    1. Class weights: Give higher weight to minority classes
       - sklearn: class_weight='balanced'
       - TensorFlow/Keras: class_weight parameter in model.fit()
    2. SMOTE or oversampling for minority classes
    3. Undersampling the majority class (Severity 2)
    4. Consider binary: "severe" (3-4) vs "not severe" (1-2)
    5. Focal loss — designed for class imbalance

    For evaluation: Use weighted F1, not just accuracy.
    Weighted F1 accounts for class imbalance by weighting each class by its support.
    """
    print("Severity Distribution:")
    print(df['Severity'].value_counts().sort_index())
    print(f"\nClass ratios:")
    print(df['Severity'].value_counts(normalize=True).sort_index().round(3))

# =============================================================================
# HINT 10: Geographic Feature Engineering
# =============================================================================
def create_geographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Location matters for accident severity prediction.

    Feature ideas:
    1. State-level patterns (some states have more severe accidents)
    2. Urban vs. rural (can infer from city population or zip code)
    3. Latitude as a proxy for climate (northern = more ice/snow)
    4. Distance from nearest airport (proxy for traffic volume)
    5. Cluster analysis on lat/lng to find accident hotspots

    Warning: Don't use raw lat/lng as features — they're too specific
    and lead to overfitting. Instead, bin them or use for clustering.
    """
    # State-level average severity (target encoding — be careful of leakage!)
    # Only compute on training data, then apply to test

    # Latitude bins (rough climate proxy)
    if 'Start_Lat' in df.columns:
        df['lat_bin'] = pd.cut(df['Start_Lat'], bins=10, labels=False)

    return df

# =============================================================================
# Find top 5 zipcode in each region and group the rest into "other" category
# =============================================================================
def create_zipcode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group zip codes within available region columns.

    Robust to:
    - missing region dummy columns
    - missing Zipcode column
    - tiny prediction batches
    """
    df = df.copy()

    # If Zipcode is missing, nothing to do
    if 'Zipcode' not in df.columns:
        return df

    # Only use region columns that actually exist
    expected_region_cols = [
        'region_Midwest',
        'region_Northeast',
        'region_South',
        'region_West',
        'region_Other'
    ]
    region_cols = [col for col in expected_region_cols if col in df.columns]

    # If no region dummy columns exist, fall back to a generic region label
    if not region_cols:
        temp_region = pd.Series(['region_unknown'] * len(df), index=df.index)
    else:
        # For rows where all region flags are 0, assign region_unknown
        region_matrix = df[region_cols].fillna(0)
        temp_region = region_matrix.idxmax(axis=1)
        no_region_mask = region_matrix.sum(axis=1) == 0
        temp_region.loc[no_region_mask] = 'region_unknown'

    # Normalize zipcode to string
    zipcode_str = df['Zipcode'].astype(str).fillna('missing')

    # For each region, keep top 5 zipcodes; group others into "other"
    is_top_5 = zipcode_str.groupby(temp_region).transform(
        lambda x: x.isin(x.value_counts().nlargest(5).index)
    )

    df['Zip_Grouped'] = temp_region + "_" + zipcode_str
    df.loc[~is_top_5, 'Zip_Grouped'] = temp_region + "_other"

    # One-hot encode
    df = pd.get_dummies(df, columns=['Zip_Grouped'], prefix='Zip', dtype=int)

    return df

# ==================================================================================================
# Find top 20 cities and counties and group the rest into "Other" category, then one-hot encode
# ==================================================================================================
def encode_top_geo_features(df, columns=['City', 'County']):
    for col in columns:
        # 1. Find the top 20 values for the current column
        top_20 = df[col].value_counts().nlargest(20).index
        
        # 2. Rename anything not in the top 20 to 'Other'
        df[col] = df[col].where(df[col].isin(top_20), 'Other')
    
    # 3. One-Hot Encode both columns at once
    # prefix=['City', 'Cty'] keeps the new column names clean
    df = pd.get_dummies(df, columns=columns, prefix=['City', 'Cty'])
    
    return df

# ==================================================================================================
# A WAY TO COMBINE WEATHER CONDITIONS INTO A SINGLE RISK SCORE
# ==================================================================================================
def dangerous_conditions_score(df): 
    df = df.copy()
    
    # Apply the scoring function to every row
    print("Engineering DangerousScore...")
    df['DangerousScore'] = df.apply(calculate_dangerous_score, axis=1)
    
    # NOW drop the columns once the scoring is done
    cols_to_drop = [
        'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 
        'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 
        'Weather_Condition', 'Sunrise_Sunset', 'Astronomical_Twilight'
    ]
    
    # Only drop columns that actually exist in the dataframe
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_drops)
    
    return df

def calculate_dangerous_score(row): 
    score = 0
    # Get weather text and handle potential missing values
    weather = str(row.get('Weather_Condition', '')).strip().lower()

    # Visibility
    visibility = row.get('Visibility(mi)')
    if pd.notna(visibility):
        if visibility < 1: score += 3
        elif visibility < 3: score += 2
        elif visibility < 5: score += 1

    # Precipitation
    precip = row.get('Precipitation(in)')
    if pd.notna(precip):
        if precip > 0.3: score += 2
        elif precip > 0: score += 1
    
    # Temperature / Wind Chill
    temp = row.get('Temperature(F)')
    wind_chill = row.get('Wind_Chill(F)')
    effective_temp = wind_chill if pd.notna(wind_chill) else temp

    if pd.notna(effective_temp):
        if effective_temp < 32: score += 2   # freezing
        elif effective_temp > 100: score += 1 # extreme heat

    # Wind speed
    wind = row.get('Wind_Speed(mph)')
    if pd.notna(wind):
        if wind > 40: score += 2
        elif wind > 25: score += 1

    # Darkness
    if row.get('Sunrise_Sunset') == 'Night': score += 1
    if row.get('Astronomical_Twilight') == 'Night': score += 1

    # Weather text categories
    if any(term in weather for term in ['tornado', 'thunderstorm', 'hail', 'squalls']):
        score += 3
    elif any(term in weather for term in ['freezing', 'sleet', 'ice', 'wintry']):
        score += 3
    elif any(term in weather for term in ['fog', 'mist', 'haze', 'smoke']):
        score += 2
    elif any(term in weather for term in ['rain', 'snow', 'drizzle']):
        score += 1

    return score

def engineer_road_features(df):
    """
    Groups individual boolean road features into a single 'n_road_features' count
    and a binary 'has_traffic_control' flag.
    """
    # Create a copy to avoid SettingWithCopy warnings
    df = df.copy()

    # Define all possible road features
    road_features = [
        'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
        'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
        'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop'
    ]
    
    #Filter for features that actually exist in the current dataframe
    existing = [f for f in road_features if f in df.columns]
    print(f"Aggregating {len(existing)} road features...")

    #Total road features present (Sum of booleans)
    df['n_road_features'] = df[existing].sum(axis=1)

    #Traffic control presence (Specific Subset)
    control_features = ['Traffic_Signal', 'Stop', 'Give_Way', 'Traffic_Calming']
    existing_control = [f for f in control_features if f in df.columns]
    
    # Create binary flag (1 if ANY control features are true, else 0)
    df['has_traffic_control'] = df[existing_control].any(axis=1).astype(int)

    # Remove original individual road features to reduce dimensionality
    # This helps models like Random Forest focus on the aggregated signal
    df = df.drop(columns=existing)
    
    print(f"Feature engineering complete. New columns: ['n_road_features', 'has_traffic_control']")
    return df
