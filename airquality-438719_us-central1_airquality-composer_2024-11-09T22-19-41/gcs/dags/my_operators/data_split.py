import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging
from google.cloud import storage
import io

# Configure logging for anomaly detection
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataSplitter:
    def __init__(self, bucket_name, input_file_path):
        self.bucket_name = bucket_name
        self.input_file_path = input_file_path
        self.dataframe = None
        self.train_df = None
        self.test_df = None

    def load_pickle(self):
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        blob = bucket.blob(self.input_file_path)

        # Download the pickle file from GCS
        if not blob.exists():
            logger.error(f"Pickle file '{self.input_file_path}' does not exist in bucket '{self.bucket_name}'.")
            return ["Pickle file not found"]

        # Read the pickle file directly into a DataFrame using BytesIO
        file_bytes = blob.download_as_bytes()  # Download file as bytes
        file_like_object = io.BytesIO(file_bytes)  # Convert bytes into a file-like object
        self.dataframe = pd.read_pickle(file_like_object)  # Load dataframe from the file-like object
        logger.info(f"Loaded data from {self.input_file_path} in bucket '{self.bucket_name}'.")
        
        if self.dataframe.empty:
            logger.warning("Loaded DataFrame is empty.")
            return ["Loaded DataFrame is empty"]
        return []

    def detect_anomalies(self):
        anomalies = []
        if self.dataframe is None:
            anomalies.append("No DataFrame loaded.")
            logger.error("No DataFrame loaded.")
            return anomalies
        
        # Check for essential columns
        required_columns = ["date", "parameter", "value"]
        missing_columns = [col for col in required_columns if col not in self.dataframe.columns]
        if missing_columns:
            anomaly = f"Missing columns in DataFrame: {', '.join(missing_columns)}"
            logger.error(anomaly)
            anomalies.append(anomaly)
        
        # Check for empty DataFrame after loading
        if self.dataframe.empty:
            anomalies.append("DataFrame is empty.")
            logger.error("DataFrame is empty.")
        
        return anomalies

    def split_data(self, test_size=0.2, random_state=42):
        if self.dataframe is None:
            logger.error("No DataFrame to split. Please load the pickle file first.")
            return ["No DataFrame to split"]

        self.train_df, self.test_df = train_test_split(self.dataframe, test_size=test_size, random_state=random_state)
        if self.train_df.empty or self.test_df.empty:
            logger.warning("Split resulted in an empty train or test set.")
            return ["Empty train or test set after split"]
        
        logger.info(f"Data split into training (size={len(self.train_df)}) and testing (size={len(self.test_df)}) sets.")
        return []

    def save_as_pickle(self, train_output_path, test_output_path):
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)

        if self.train_df is None or self.test_df is None:
            logger.error("No split data to save. Please split the data first.")
            return ["No split data to save"]
        
        # Create BytesIO object to save the DataFrame as a byte stream
        train_bytes = io.BytesIO()
        self.train_df.to_pickle(train_bytes)
        train_bytes.seek(0)  # Rewind the BytesIO object to the beginning

        test_bytes = io.BytesIO()
        self.test_df.to_pickle(test_bytes)
        test_bytes.seek(0)

        # Upload training DataFrame to GCS
        train_blob = bucket.blob(train_output_path)
        train_blob.upload_from_file(train_bytes, content_type='application/octet-stream')
        logger.info(f"Training DataFrame saved as '{train_output_path}' in bucket '{self.bucket_name}'.")

        # Upload testing DataFrame to GCS
        test_blob = bucket.blob(test_output_path)
        test_blob.upload_from_file(test_bytes, content_type='application/octet-stream')
        logger.info(f"Testing DataFrame saved as '{test_output_path}' in bucket '{self.bucket_name}'.")
        
        return []

def split():
    # Load environment variables for bucket names and file paths
    bucket_name = os.environ.get("DATA_SPLIT_BUCKET_NAME")
    input_file_path = os.environ.get("DATA_SPLIT_INPUT_FILE_PATH")
    train_output_file_path = os.environ.get("DATA_SPLIT_TRAIN_OUTPUT_FILE_PATH")
    test_output_file_path = os.environ.get("DATA_SPLIT_TEST_OUTPUT_FILE_PATH")

    data_splitter = DataSplitter(bucket_name, input_file_path)
    anomalies = []

    # Step 1: Load and check for anomalies
    anomalies.extend(data_splitter.load_pickle())
    anomalies.extend(data_splitter.detect_anomalies())

    # Step 2: Split data and check for anomalies
    if not anomalies:
        anomalies.extend(data_splitter.split_data(test_size=0.2, random_state=42))
    
    # Step 3: Save split data and check for anomalies
    if not anomalies:
        anomalies.extend(data_splitter.save_as_pickle(train_output_file_path, test_output_file_path))

    if anomalies:
        logger.error(f"Anomalies detected during splitting: {anomalies}")
    else:
        logger.info("Data splitting completed successfully with no anomalies.")
    
    return anomalies

if __name__ == "__main__":
    detected_anomalies = split()
    if detected_anomalies:
        logger.error(f"Detected Anomalies: {detected_anomalies}")
    else:
        logger.info("No anomalies detected. Process completed successfully.")
