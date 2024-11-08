import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from google.cloud import storage
from io import BytesIO
import os

# Configure logging for anomaly detection
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataSplitter:
    def __init__(self, gcs_bucket_name, pickle_file_path):
        self.gcs_bucket_name = gcs_bucket_name
        self.pickle_file_path = pickle_file_path
        self.dataframe = None
        self.train_df = None
        self.test_df = None

    def load_pickle(self):
        """Load a pickle file from Google Cloud Storage."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.gcs_bucket_name)
        blob = bucket.blob(self.pickle_file_path)
        
        try:
            data = blob.download_as_bytes()
            self.dataframe = pd.read_pickle(BytesIO(data))
            logger.info(f"Loaded data from GCS at '{self.pickle_file_path}'.")
        except Exception as e:
            logger.error(f"Error loading data from GCS: {e}")
            return ["Error loading data"]
        
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
        
        # Check for essential columns (if known in advance)
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
        if self.train_df is None or self.test_df is None:
            logger.error("No split data to save. Please split the data first.")
            return ["No split data to save"]
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.gcs_bucket_name)

        # Save train DataFrame
        train_blob = bucket.blob(train_output_path)
        train_buffer = BytesIO()
        self.train_df.to_pickle(train_buffer)
        train_buffer.seek(0)
        train_blob.upload_from_file(train_buffer, content_type='application/octet-stream')
        logger.info(f"Training DataFrame saved to GCS at '{train_output_path}'.")

        # Save test DataFrame
        test_blob = bucket.blob(test_output_path)
        test_buffer = BytesIO()
        self.test_df.to_pickle(test_buffer)
        test_buffer.seek(0)
        test_blob.upload_from_file(test_buffer, content_type='application/octet-stream')
        logger.info(f"Testing DataFrame saved to GCS at '{test_output_path}'.")

def perform_data_split(bucket_name, input_pickle_file_path, output_train_file_path, output_test_file_path, test_size=0.2, random_state=42):
    """Perform data splitting and save results to GCS."""
    data_splitter = DataSplitter(bucket_name, input_pickle_file_path)
    anomalies = []

    # Step 1: Load and check for anomalies
    anomalies.extend(data_splitter.load_pickle())
    anomalies.extend(data_splitter.detect_anomalies())

    # Step 2: Split data and check for anomalies
    if not anomalies:
        anomalies.extend(data_splitter.split_data(test_size=test_size, random_state=random_state))
    
    # Step 3: Save split data and check for anomalies
    if not anomalies:
        data_splitter.save_as_pickle(output_train_file_path, output_test_file_path)

    if anomalies:
        logger.error(f"Anomalies detected during splitting: {anomalies}")
    else:
        logger.info("Data splitting completed successfully with no anomalies.")
    
    return anomalies

# This function should be called in the context of a DAG in Cloud Composer
if __name__ == "__main__":
    BUCKET_NAME = 'your_bucket_name'  # Replace with your GCS bucket name
    INPUT_PICKLE_FILE_PATH = 'dags/DataPreprocessing/src/data_store_pkl_files/resampled_data.pkl'
    OUTPUT_TRAIN_FILE_PATH = 'dags/DataPreprocessing/src/data_store_pkl_files/train_data/train_data.pkl'
    OUTPUT_TEST_FILE_PATH = 'dags/DataPreprocessing/src/data_store_pkl_files/test_data/test_data.pkl'
    
    perform_data_split(BUCKET_NAME, INPUT_PICKLE_FILE_PATH, OUTPUT_TRAIN_FILE_PATH, OUTPUT_TEST_FILE_PATH)
