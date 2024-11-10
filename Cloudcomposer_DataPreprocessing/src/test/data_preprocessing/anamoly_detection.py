import pandas as pd
import numpy as np
import os
import logging
from google.cloud import storage
from io import BytesIO

# Configure logging for anomaly detection
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.anomalies = []
        self.outlier_count = 0
        self.negative_value_count = 0

    def load_data(self):
        try:
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
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def handle_outliers(self, column_name='pm25'):
        if column_name not in self.data.columns:
            anomaly = f"'{column_name}' column not found in the DataFrame."
            logger.error(anomaly)
            self.anomalies.append(anomaly)
            return

        Q1 = self.data[column_name].quantile(0.25)
        Q3 = self.data[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count outliers
        outliers = self.data[(self.data[column_name] < lower_bound) | (self.data[column_name] > upper_bound)]
        self.outlier_count = len(outliers)
        if self.outlier_count > 0:
            self.anomalies.append(f"{self.outlier_count} outliers detected in '{column_name}' column.")

        # Replace outliers with median
        column_median = self.data[column_name].median()
        self.data.loc[(self.data[column_name] < lower_bound) | (self.data[column_name] > upper_bound), column_name] = column_median
        logger.info(f"Outliers in '{column_name}' replaced with median value {column_median}.")

    def replace_negative_with_zero(self, column_name='pm25'):
        if column_name not in self.data.columns:
            anomaly = f"'{column_name}' column not found in the DataFrame."
            logger.error(anomaly)
            self.anomalies.append(anomaly)
            return

        # Count negative values
        negative_values = self.data[self.data[column_name] < 0]
        self.negative_value_count = len(negative_values)
        if self.negative_value_count > 0:
            self.anomalies.append(f"{self.negative_value_count} negative values detected in '{column_name}' column.")

        # Replace negative values with 0
        self.data[column_name] = self.data[column_name].clip(lower=0)
        logger.info(f"Negative values in '{column_name}' replaced with 0.")

    def save_as_pickle(self, output_path):
        if self.data is not None:
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
            logger.info(f"Cleaned DataFrame saved as '{output_path}'.")
        else:
            logger.error("No data available to save. Please load and clean the data first.")
            self.anomalies.append("Data not loaded or processed; nothing to save.")

def anamoly_detection_val():
    # Path to the input pickle file and output pickle file
    file_path = os.environ.get("TEST_DATA_ANMLY_INPUT")
    output_pickle_file = os.environ.get("TEST_DATA_ANMLY_OUTPUT")
    
    cleaner = DataCleaner(file_path)
    cleaner.load_data()
    cleaner.handle_outliers(column_name='pm25')
    cleaner.replace_negative_with_zero(column_name='pm25')
    cleaner.save_as_pickle(output_pickle_file)

    # Log summary of detected anomalies
    logger.info(f"Total anomalies: {cleaner.anomalies}")
    logger.info(f"Total outliers replaced: {cleaner.outlier_count}")
    logger.info(f"Total negative values replaced: {cleaner.negative_value_count}")
    
    return cleaner.anomalies  # Return list of detected anomalies

if __name__ == "__main__":
    detected_anomalies = anamoly_detection_val()
    if detected_anomalies:
        logger.error(f"Detected Anomalies: {detected_anomalies}")
    else:
        logger.info("No anomalies detected. Process completed successfully.")
