#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import json
import logging
from google.cloud import storage
from io import BytesIO
from fairlearn.metrics import MetricFrame
from sklearn.utils import resample
import os

# Set up logging configuration
log_file_path = '/tmp/process_data_bias.log'  # Use a temporary path for logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)

# Constants for GCS paths
BUCKET_NAME = 'your_bucket_name'  # Replace with your GCS bucket name
DATASET_FILE_PATH = 'dags/DataPreprocessing/src/data_store_pkl_files/air_pollution.pkl'
RESULTS_FILE_PATH = 'dags/resampled_data.json'

class PM25Analysis:
    def __init__(self, data_source):
        if isinstance(data_source, str):
            self.filepath = data_source
            self.data_pm25 = None
        elif isinstance(data_source, pd.DataFrame):
            self.data_pm25 = data_source
            self.filepath = None
        else:
            raise ValueError("data_source should be a file path or a DataFrame")

        self.results = {}

    def load_and_filter_data(self):
        if self.data_pm25 is None and self.filepath is not None:
            data = pd.read_pickle(self.filepath)
            self.data_pm25 = data[data['parameter'] == 'pm25']
        elif self.data_pm25 is not None:
            self.data_pm25 = self.data_pm25[self.data_pm25['parameter'] == 'pm25']
        else:
            raise ValueError("No data source available.")
        return self.data_pm25

    def fairlearn_bias_check_with_flag(self, sensitive_feature, threshold=0.2):
        if sensitive_feature not in self.data_pm25.columns:
            raise ValueError(f"{sensitive_feature} not found in data columns.")
        
        metric_frame = MetricFrame(
            metrics={'count': lambda x, y: len(x)},
            y_true=self.data_pm25['value'],
            y_pred=self.data_pm25['value'],
            sensitive_features=self.data_pm25[sensitive_feature]
        )
        
        avg_count = metric_frame.by_group['count'].mean()
        biased_groups_df = metric_frame.by_group['count'].to_frame(name='count')
        biased_groups_df['bias_flag'] = ((biased_groups_df['count'] > avg_count * (1 + threshold)) |
                                         (biased_groups_df['count'] < avg_count * (1 - threshold)))
        
        self.results[f'fairlearn_bias_{sensitive_feature}_with_bias_flag'] = biased_groups_df
        logger.info("Bias check completed for sensitive feature: %s", sensitive_feature)
        return biased_groups_df

    def resample_biased_months(self, target_feature='month'):
        if f'fairlearn_bias_{target_feature}_with_bias_flag' not in self.results:
            raise ValueError(f"Run fairlearn_bias_check_with_flag with {target_feature} before resampling.")
        
        biased_groups_df = self.results[f'fairlearn_bias_{target_feature}_with_bias_flag']
        unbiased_counts = biased_groups_df[~biased_groups_df['bias_flag']]['count']
        target_count = int(unbiased_counts.mean())
        
        biased_months = biased_groups_df[biased_groups_df['bias_flag']].index
        biased_data = self.data_pm25[self.data_pm25[target_feature].isin(biased_months)]
        unbiased_data = self.data_pm25[~self.data_pm25[target_feature].isin(biased_months)]
        
        resampled_data = []
        
        for month in biased_months:
            month_data = biased_data[biased_data[target_feature] == month]
            if len(month_data) > target_count:
                month_data_resampled = resample(month_data, replace=False, n_samples=target_count, random_state=42)
            else:
                month_data_resampled = resample(month_data, replace=True, n_samples=target_count, random_state=42)
            resampled_data.append(month_data_resampled)
        
        resampled_data = pd.concat(resampled_data + [unbiased_data], ignore_index=True)
        logger.info("Resampling completed. Total records after resampling: %d", len(resampled_data))

        return resampled_data

def load_pickle_from_gcs(bucket_name, file_path):
    """Load a pickled DataFrame from Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    
    data = blob.download_as_bytes()
    df = pd.read_pickle(BytesIO(data))
    logger.info("Loaded DataFrame from GCS at '%s'.", file_path)
    return df

def save_results_to_gcs(bucket_name, data, file_path):
    """Save results to a JSON file in Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Save to GCS
    blob = bucket.blob(file_path)
    blob.upload_from_string(json.dumps(data, indent=4), content_type='application/json')
    logger.info("Results saved to GCS at '%s'.", file_path)

def bias_main(bucket_name, input_pickle_file, sensitive_feature=None, threshold=0.2):
    """Perform PM2.5 analysis and save results to GCS."""
    data_pm25 = load_pickle_from_gcs(bucket_name, input_pickle_file)
    
    analysis = PM25Analysis(data_pm25)
    analysis.load_and_filter_data()
    
    if sensitive_feature:
        analysis.fairlearn_bias_check_with_flag(sensitive_feature, threshold)
        resampled_data = analysis.resample_biased_months(target_feature=sensitive_feature)
        save_results_to_gcs(bucket_name, resampled_data.to_dict(), RESULTS_FILE_PATH)

# This function should be called in the context of a DAG in Cloud Composer
if __name__ == "__main__":
    bias_main(BUCKET_NAME, DATASET_FILE_PATH, sensitive_feature='month', threshold=0.2)
