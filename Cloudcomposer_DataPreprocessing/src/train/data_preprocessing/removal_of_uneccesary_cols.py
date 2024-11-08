import pandas as pd
import logging
from google.cloud import storage
from io import BytesIO

# Configure logging for anomaly detection
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, gcs_bucket_name, file_path):
        self.gcs_bucket_name = gcs_bucket_name
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Load data from Google Cloud Storage."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.gcs_bucket_name)
        blob = bucket.blob(self.file_path)

        try:
            data = blob.download_as_bytes()
            self.data = pd.read_pickle(BytesIO(data))
            logger.info(f"Data loaded from GCS at '{self.file_path}'.")
        except Exception as e:
            logger.error(f"Error loading data from GCS: {e}")
            return ["Error loading data"]
        
        if self.data.empty:
            logger.warning("Loaded DataFrame is empty.")
            return ["Loaded DataFrame is empty"]
        return []

    def drop_columns(self, columns_to_drop):
        if self.data is None:
            logger.error("Data is not loaded. Cannot drop columns.")
            return ["Data not loaded"]
        
        # Check which columns are present
        columns_present = [col for col in columns_to_drop if col in self.data.columns]

        if not columns_present:
            logger.warning("No specified columns found in DataFrame to drop.")
            return ["No columns to drop"]

        # Drop specified columns
        self.data.drop(columns=columns_present, inplace=True)
        logger.info(f"Dropped columns: {', '.join(columns_present)}.")

    def save_data(self, output_path):
        """Save the cleaned data to Google Cloud Storage."""
        if self.data is None:
            logger.error("No data to save. Please load and clean data first.")
            return ["No data to save"]

        storage_client = storage.Client()
        bucket = storage_client.bucket(self.gcs_bucket_name)
        output_blob = bucket.blob(output_path)

        # Save cleaned DataFrame
        buffer = BytesIO()
        self.data.to_pickle(buffer)
        buffer.seek(0)
        output_blob.upload_from_file(buffer, content_type='application/octet-stream')
        logger.info(f"Cleaned data saved to GCS at '{output_path}'.")

def remove_uneccesary_cols(bucket_name, input_file_path, output_file_path, columns_to_drop):
    """Perform data loading, column removal, and saving results to GCS."""
    cleaner = DataCleaner(bucket_name, input_file_path)

    anomalies = cleaner.load_data()
    if anomalies:
        logger.error(f"Errors encountered while loading data: {anomalies}")
        return

    cleaner.drop_columns(columns_to_drop)

    cleaner.save_data(output_file_path)

# This function should be called in the context of a DAG in Cloud Composer
if __name__ == "__main__":
    BUCKET_NAME = 'your_bucket_name'  # Replace with your GCS bucket name
    INPUT_FILE_PATH = 'processed_data/stacked_air_pollution.pkl'
    OUTPUT_FILE_PATH = 'cleaned_data/cleaned_air_pollution.pkl'
    COLUMNS_TO_DROP = ['unnecessary_column1', 'unnecessary_column2']  # Replace with actual columns to drop

    remove_uneccesary_cols(BUCKET_NAME, INPUT_FILE_PATH, OUTPUT_FILE_PATH, COLUMNS_TO_DROP)
