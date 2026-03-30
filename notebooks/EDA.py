import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Add project root to sys.path
root_path = Path.cwd().parent 
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

# Import function from pipeline
from pipelines.data_pipeline import load_raw_data, clean_data, accident_engineer_features, complaints_engineer_features, create_temporal_features
from pipelines.data_pipeline import generate_hourly_heatmap, generate_accident_map # functions to create maps

#Load the City Traffic Accident Database
df_City_Traffic = load_raw_data("city_traffic_accidents.csv")
df_Complaints= load_raw_data("urbanpulse_311_complaints.csv")

#Clean and engineer features for the City Traffic Accident Database
df_City_Traffic = clean_data(df_City_Traffic)                       #Clean the data (handle missing values, convert data types, etc.)
df_City_Traffic = accident_engineer_features(df_City_Traffic)       #Engineer features specific to traffic accidents (e.g., severity, weather conditions, etc.)


#Clean and engineer features for the 311 Complaints Database
df_Complaints= clean_data(df_Complaints)                            #Clean the data (handle missing values, convert data types, etc.)
df_Complaints = complaints_engineer_features(df_Complaints)         #Engineer features specific to 311 complaints (e.g., complaint type, resolution time, etc.)

#Generate the heatmap and accident map for City Traffic Accident
generate_hourly_heatmap(df_City_Traffic)                            #Generate a heatmap to visualize the density of accidents over time and location
generate_accident_map(df_City_Traffic)                              #Generate a map to visualize the locations of accidents and their severity