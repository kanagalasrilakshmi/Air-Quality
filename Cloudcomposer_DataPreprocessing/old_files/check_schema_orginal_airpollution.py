import pandas as pd
import numpy as np
import json
import logging
from google.cloud import storage
from io import BytesIO

# Set up logging configuration to log to a file and console
log_file_path = '/tmp/process_check_schema_air_pollution.log'  # Use a temporary path
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)

# Constants for file paths
BUCKET_NAME = 'airquality-mlops-rg'  # Replace with your GCS bucket name
SCHEMA_FILE_PATH = 'schema/custom_schema_generated_from_api.json'
DATASET_FILE_PATH = 'processed_data/stacked_air_pollution.pkl'
STATS_FILE_PATH = 'schema/air_pollution_stats.json'

# Function to load a pickled DataFrame from GCS
def load_pickle_from_gcs(bucket_name, file_path):
    """Load a pickled DataFrame from Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    
    data = blob.download_as_bytes()
    df = pd.read_pickle(BytesIO(data))
    logger.info("Loaded DataFrame from GCS at '%s'.", file_path)
    return df

# Function to save data (schema or statistics) to a JSON file in GCS
def save_to_gcs(bucket_name, data, file_path):
    """Save data (schema or statistics) to a JSON file in Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Convert the data for serialization
    data_to_save = convert_data_for_serialization(data)

    # Save to GCS
    blob = bucket.blob(file_path)
    blob.upload_from_string(json.dumps(data_to_save, indent=4), content_type='application/json')
    logger.info("Data saved to GCS at '%s'.", file_path)

def convert_data_for_serialization(data):
    """Convert pandas DataFrame/Series to a JSON serializable format."""
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return data.to_dict()
    elif isinstance(data, dict):
        return {k: convert_data_for_serialization(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_data_for_serialization(i) for i in data]
    return data

# Function to generate schema based on dataset structure
def generate_schema(data):
    logger.info("Generating schema for the dataset.")
    # Check if 'date' and 'parameter' columns are present
    required_columns = ['date', 'parameter']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.error("Missing required columns: %s", missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    
    # Schema generation based on data structure
    schema = {"columns": []}
    for column_name, dtype in data.dtypes.items():
        column_info = {
            "name": column_name,
            "type": dtype.name,
            "required": not data[column_name].isnull().any()  # True if no missing values
        }

        # Adding specific constraints
        if column_name == "date":
            column_info["format"] = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{2}:\d{2}$"
        elif column_name == "parameter":
            column_info["allowed_values"] = list(data[column_name].unique())
        
        schema["columns"].append(column_info)
    
    return schema

# Function to generate statistics for each column
def generate_statistics(data):
    logger.info("Generating statistics for the dataset.")
    stats = {}
    for column in data.columns:
        stats[column] = {
            "mean": data[column].mean() if pd.api.types.is_numeric_dtype(data[column]) else None,
            "std_dev": data[column].std() if pd.api.types.is_numeric_dtype(data[column]) else None,
            "min": data[column].min(),
            "max": data[column].max(),
            "missing_values": data[column].isnull().sum(),
            "unique_values": len(data[column].unique())
        }
    return stats

# Main function to load the dataset, generate schema and statistics, and save them
def generate_schema_and_statistics():
    data = load_pickle_from_gcs(BUCKET_NAME, DATASET_FILE_PATH)

    if isinstance(data, pd.Series):
        data = data.to_frame()
    elif not isinstance(data, pd.DataFrame):
        raise ValueError("Loaded data is not a DataFrame or Series.")

    # Generate schema and statistics
    schema = generate_schema(data)
    stats = generate_statistics(data)

    # Save schema and statistics to GCS
    save_to_gcs(BUCKET_NAME, schema, SCHEMA_FILE_PATH)
    save_to_gcs(BUCKET_NAME, stats, STATS_FILE_PATH)

# This function should be called in the context of a DAG in Cloud Composer
if __name__ == "__main__":
    generate_schema_and_statistics()
