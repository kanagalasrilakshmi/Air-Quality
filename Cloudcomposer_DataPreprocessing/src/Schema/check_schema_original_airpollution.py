import pandas as pd
import numpy as np
import os
import json
import logging
from google.cloud import storage

# Set up logging configuration to log to a file and console
log_file_path = os.path.join(os.getcwd(), 'process_check_schema_air_pollution.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),   # Log to a file
        logging.StreamHandler()               # Also log to console
    ]
)
logger = logging.getLogger(__name__)

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

        # Adding specific constraints (e.g., date format for "date" column, allowed values for "parameter" column)
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

# Save data (schema or statistics) to a JSON file in GCS
def save_to_gcs(bucket_name, data, file_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Convert the data for serialization
    data_to_save = convert_types(data)

    # Save to GCS
    blob = bucket.blob(file_path)
    blob.upload_from_string(json.dumps(data_to_save, indent=4), content_type='application/json')
    logger.info("Data saved to GCS at '%s'.", file_path)

def convert_types(obj):
    """Convert pandas DataFrame/Series to a JSON serializable format."""
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(i) for i in obj]
    return obj

# Load the dataset, generate schema and statistics, and save them
def main_generate_schema_and_statistics():
    # Load environment variables for bucket names and file paths
    bucket_name = os.environ.get("SCHEMA_STATS_BUCKET_NAME")
    dataset_file_path = os.environ.get("SCHEMA_STATS_INPUT_FILE_PATH")
    schema_file_path = os.environ.get("SCHEMA_FILE_OUTPUT_PATH")
    stats_file_path = os.environ.get("STATS_FILE_OUTPUT_PATH")

    # Read the dataset from GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(dataset_file_path)

    # Check if the dataset exists
    if not blob.exists():
        logger.error(f"Dataset '{dataset_file_path}' does not exist in bucket '{bucket_name}'.")
        raise FileNotFoundError(f"Dataset '{dataset_file_path}' not found.")

    # Load dataset into a DataFrame
    data = pd.read_pickle(blob.download_as_bytes())
    logger.info("Loaded dataset from %s", dataset_file_path)

    # Generate schema and statistics
    schema = generate_schema(data)
    stats = generate_statistics(data)

    # Save schema and statistics to GCS
    save_to_gcs(bucket_name, schema, schema_file_path)
    save_to_gcs(bucket_name, stats, stats_file_path)

# Run the main function
if __name__ == "__main__":
    main_generate_schema_and_statistics()
