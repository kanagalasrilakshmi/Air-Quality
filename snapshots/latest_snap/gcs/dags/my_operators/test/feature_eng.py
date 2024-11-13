import pandas as pd
import numpy as np
from scipy import stats
import os
from google.cloud import storage
from io import BytesIO
import logging

# Configure logging for anomaly detection
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataFeatureEngineer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.fitting_lambda = None

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

    def handle_skewness(self, column_name='pm25'):
        skewness = self.data[column_name].skew()
        logger.info(f'Original Skewness: {skewness}')
        if np.abs(skewness) < 0.5:
            return column_name
        else:
            self.data[f'{column_name}_log'] = np.log1p(self.data[column_name])
            log_skewness = self.data[f'{column_name}_log'].skew()
            logger.info(f'Log Transformed Skewness: {log_skewness}')

            self.data[f'{column_name}_boxcox'], self.fitting_lambda = stats.boxcox(self.data[column_name] + 1)
            boxcox_skewness = self.data[f'{column_name}_boxcox'].skew()
            logger.info(f'Box-Cox Transformed Skewness: {boxcox_skewness}')

            if abs(boxcox_skewness) < abs(log_skewness):
                self.data.drop(columns=[f'{column_name}_log'], inplace=True)
                logger.info("Choosing Box-Cox transformed column.")
                return f'{column_name}_boxcox'
            else:
                logger.info("Choosing Log transformed column.")
                self.data.drop(columns=[f'{column_name}_boxcox'], inplace=True)
                return f'{column_name}_log'

    def feature_engineering(self, chosen_column):
        logger.info("Starting feature engineering.")
        # Create lag features
        for lag in range(1, 6):  # Creates lag_1 to lag_5
            self.data[f'lag_{lag}'] = self.data[chosen_column].shift(lag)

        self.data['rolling_mean_3'] = self.data[chosen_column].rolling(window=3).mean()
        self.data['rolling_mean_6'] = self.data[chosen_column].rolling(window=6).mean()
        self.data['rolling_mean_24'] = self.data[chosen_column].rolling(window=24).mean()
        self.data['rolling_std_3'] = self.data[chosen_column].rolling(window=3).std()
        self.data['rolling_std_6'] = self.data[chosen_column].rolling(window=6).std()
        self.data['rolling_std_24'] = self.data[chosen_column].rolling(window=24).std()
        self.data['ema_3'] = self.data[chosen_column].ewm(span=3, adjust=False).mean()
        self.data['ema_6'] = self.data[chosen_column].ewm(span=6, adjust=False).mean()
        self.data['ema_24'] = self.data[chosen_column].ewm(span=24, adjust=False).mean()
        self.data['diff_1'] = self.data[chosen_column].diff(1)
        self.data['diff_2'] = self.data[chosen_column].diff(2)
        self.data['hour'] = self.data.index.hour
        self.data['day_of_week'] = self.data.index.dayofweek
        self.data['day_of_year'] = self.data.index.dayofyear
        self.data['month'] = self.data.index.month
        self.data['sin_hour'] = np.sin(2 * np.pi * self.data['hour'] / 24)
        self.data['cos_hour'] = np.cos(2 * np.pi * self.data['hour'] / 24)
        self.data['sin_day_of_week'] = np.sin(2 * np.pi * self.data['day_of_week'] / 7)
        self.data['cos_day_of_week'] = np.cos(2 * np.pi * self.data['day_of_week'] / 7)
        self.data.dropna(inplace=True)
        logger.info("Feature engineering completed and NaN values dropped.")

    def save_as_pickle(self, output_path):
        if self.data is not None:
            # Save the processed DataFrame to GCS
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
        else:
            logger.error("No data available to save. Please load and process the data first.")

def feature_engineering():
    # Path to the input pickle file and output pickle file
    file_path = os.environ.get("TEST_DATA_FEA_ENG_INPUT")
    output_pickle_file = os.environ.get("TEST_DATA_FEA_ENG_OUTPUT")

    engineer = DataFeatureEngineer(file_path)
    engineer.load_data()
    chosen_column = engineer.handle_skewness(column_name='pm25')
    engineer.feature_engineering(chosen_column)
    engineer.save_as_pickle(output_pickle_file)

if __name__ == "__main__":
    feature_engineering()
