import pandas as pd
import numpy as np
import os
import logging
from google.cloud import storage
from io import BytesIO

# Configure logging for anomaly detection
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        # Load data from GCS
        storage_client = storage.Client()
        bucket_name = os.environ.get("DATA_BUCKET_NAME")
        blob_name = os.path.join(self.file_path)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            logger.error(f"File '{self.file_path}' does not exist in bucket '{bucket_name}'.")
            return ["File does not exist"]

        # Download the blob and load it into a DataFrame
        data = blob.download_as_bytes()
        self.data = pd.read_pickle(BytesIO(data))
        logger.info(f"Data loaded from {self.file_path}.")
        return []

    def handle_missing_values(self):
        anomalies = []
        
        # Check for any missing values in the dataset
        missing_summary = self.data.isnull().sum()
        total_missing = missing_summary.sum()
        
        if total_missing > 0:
            logger.warning(f"Missing values found: {missing_summary[missing_summary > 0].to_dict()}")
            anomalies.append(f"Total missing values: {total_missing}")
            
            if 'pm25' in self.data.columns and self.data['pm25'].isnull().any():
                self.data['pm25'] = self.data['pm25'].interpolate(method='linear')  # Apply linear interpolation
                logger.info("'pm25' missing values interpolated.")
            else:
                logger.warning("'pm25' column not found for interpolation or no missing values in 'pm25'.")
        else:
            logger.info("No missing values found.")
        
        return anomalies

    def save_as_pickle(self, output_path):
        anomalies = []
        
        # Check if data is available to save
        if self.data is None:
            anomaly = "No data available to save. Please load and process the data first."
            logger.error(anomaly)
            anomalies.append(anomaly)
            return anomalies

        # Save the cleaned DataFrame to GCS
        storage_client = storage.Client()
        bucket_name = os.environ.get("DATA_BUCKET_NAME")
        output_blob_name = os.path.join(output_path)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(output_blob_name)

        # Write DataFrame to a bytes buffer and upload
        buffer = BytesIO()
        self.data.to_pickle(buffer)
        buffer.seek(0)
        blob.upload_from_file(buffer, content_type='application/octet-stream')
        logger.info(f"Processed DataFrame saved as '{output_path}'.")
        return []

def handle_missing_vals():
    # Paths for input and output pickle files
    file_path = os.environ.get("TEST_DATA_MISS_VAL_INPUT")
    output_pickle_file = os.environ.get("TEST_DATA_MISS_VAL_OUTPUT")
    processor = DataProcessor(file_path)
    
    anomalies = []
    # Step 1: Load data
    anomalies.extend(processor.load_data())

    # Step 2: Handle missing values
    anomalies.extend(processor.handle_missing_values())

    # Step 3: Save the cleaned DataFrame
    if not anomalies:
        anomalies.extend(processor.save_as_pickle(output_pickle_file))
        logger.info("Data cleaning process completed successfully.")
    else:
        logger.error("Anomalies detected; skipping saving the cleaned data.")
    
    return anomalies  # Return a list of all detected anomalies

if __name__ == "__main__":
    detected_anomalies = handle_missing_vals()
    if detected_anomalies:
        logger.error(f"Detected Anomalies: {detected_anomalies}")
    else:
        logger.info("No anomalies detected. Process completed successfully.")
