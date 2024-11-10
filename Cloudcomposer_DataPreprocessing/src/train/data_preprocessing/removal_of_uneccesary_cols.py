import pandas as pd
import os
import logging
from google.cloud import storage
import io  # For handling byte streams

# Configure logging for anomaly detection
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, bucket_name, file_path):
        self.bucket_name = bucket_name
        self.file_path = file_path
        self.data = None
        self.pivoted_data = None

    def load_data(self):
        # Load the data from GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        blob = bucket.blob(self.file_path)

        # Check if the file exists
        if not blob.exists():
            logger.error(f"File '{self.file_path}' does not exist in bucket '{self.bucket_name}'.")
            return ["File does not exist"]
        
        # Load the data
        self.data = pd.read_pickle(io.BytesIO(blob.download_as_bytes()))  # Use BytesIO to handle byte data
        logger.info(f"Data loaded from {self.file_path}")
        return []

    def check_column_consistency(self):
        # Check for the presence of required columns
        required_columns = {'date', 'parameter', 'value'}
        missing_columns = required_columns - set(self.data.columns)
        if missing_columns:
            anomaly = f"Missing required columns: {', '.join(missing_columns)}."
            logger.error(anomaly)
            return [anomaly]
        
        logger.info("All necessary columns are present.")
        return []

    def detect_date_anomalies(self):
        anomalies = []
        
        # Check if 'date' column exists
        if 'date' not in self.data.columns:
            anomaly = "Missing 'date' column in DataFrame."
            logger.error(anomaly)
            anomalies.append(anomaly)
            return anomalies
        
        # Convert 'date' column to datetime and handle timezone compatibility
        self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
        
        # Make sure current timestamp has the same timezone as 'date' column
        current_timestamp = pd.Timestamp.now(tz=self.data['date'].dt.tz)
        
        # Check for future dates
        future_dates = self.data[self.data['date'] > current_timestamp]
        
        if not future_dates.empty:
            anomaly = "Future dates detected in 'date' column."
            logger.warning(anomaly)
            anomalies.append(anomaly)
        
        logger.info("Date column processed successfully.")
        return anomalies

    def remove_columns(self, columns_to_remove):
        if self.data is not None:
            # Drop the specified columns
            columns_present = [col for col in columns_to_remove if col in self.data.columns]
            if columns_present:
                self.data.drop(columns=columns_present, inplace=True)
                logger.info(f"Successfully removed columns: {columns_present}")
            else:
                logger.warning(f"None of the specified columns {columns_to_remove} exist in the DataFrame.")
                return [f"None of the columns {columns_to_remove} exist in the DataFrame"]
        else:
            logger.error("No data available to remove columns. Please load the data first.")
            return ["No data loaded"]

    def save_after_column_removal(self, output_path):
        if self.data is not None:
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.bucket_name)
            blob = bucket.blob(output_path)

            # Save the DataFrame to GCS using BytesIO after column removal
            with io.BytesIO() as output:
                self.data.to_pickle(output)
                output.seek(0)  # Go to the beginning of the byte stream
                blob.upload_from_file(output, content_type='application/octet-stream')  # Upload byte stream
            logger.info(f"DataFrame saved after column removal as '{output_path}'.")
            return []
        else:
            logger.error("No data available to save. Please load and process the data first.")
            return ["No data loaded"]

def remove_uneccesary_cols():
    # Load environment variables for bucket names and file paths
    bucket_name = os.environ.get("DATA_BUCKET_NAME")
    file_path = os.environ.get("TRAIN_DATA_REMOVE_COL_INPUT")
    output_pickle_file = os.environ.get("TRAIN_DATA_REMOVE_COL_OUTPUT")

    processor = DataProcessor(bucket_name, file_path)
    
    anomalies = []
    # Step 1: Load data and check for file existence anomalies
    anomalies.extend(processor.load_data())
    
    # Step 2: Check for column consistency
    anomalies.extend(processor.check_column_consistency())

    # Step 3: Detect anomalies in the 'date' column
    anomalies.extend(processor.detect_date_anomalies())

    # Step 4: Remove specified columns if no critical anomalies are detected
    if not anomalies:
        columns_to_remove = ['co', 'no', 'no2', 'o3', 'so2']  # Example columns to remove
        anomalies.extend(processor.remove_columns(columns_to_remove))
        anomalies.extend(processor.save_after_column_removal(output_pickle_file))
        logger.info("Data processing completed successfully after column removal.")
    else:
        logger.error("Anomalies detected; skipping column removal and saving.")
    
    return anomalies  # Return list of all detected anomalies

if __name__ == "__main__":
    detected_anomalies = remove_uneccesary_cols()
    if detected_anomalies:
        logger.error(f"Detected Anomalies: {detected_anomalies}")
    else:
        logger.info("No anomalies detected. Process completed successfully.")
