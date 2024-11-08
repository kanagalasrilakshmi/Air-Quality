from airflow import DAG
from scipy import stats
from airflow.operators.python import PythonOperator
from datetime import datetime
import numpy as np
from google.cloud import storage
import io
import os
import pandas as pd
from airflow import DAG
import os
import sys
from airflow import configuration as conf
import logging

from google.cloud import storage
# from my_operators.stacks_csv import stack_csvs_to_pickle
# from my_operators.split_data import split_data
# from my_operators.pull_train import pull_train_file
# from my_operators.pivot import pivot_data_task
# from my_operators.drop_columns import clean_data
# from my_operators.check_missing import process_data
# from my_operators.check_anamoly import anomaly_detection
# from my_operators.feature_eng import feature_engineering
# from my_operators.schema_gen import data_validation
# from my_operators.data_bias import perform_analysis
# from my_operators.check_initial_schema import generate_schema_and_statistics
# from my_operators.check_output_schema import generate_output_schema_and_statistics


# # Helper function to check for new files in GCS
# def check_for_new_files(bucket_name, folder_path, **kwargs):
#     # Create a GCS client
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
    
#     # List the files in the folder
#     blobs = bucket.list_blobs(prefix=folder_path)
#     new_files = [blob.name for blob in blobs if blob.name.endswith('.csv')]
    
#     # Pass the list of files as an XCom to the next task
#     if new_files:
#         kwargs['ti'].xcom_push(key='file_list', value=new_files)
#     else:
#         raise ValueError("No new files found in the specified GCS folder")

# # Default arguments for the DAG
# default_args = {
#     'owner': 'airflow',
#     'start_date': datetime.now(),
#     'email_on_failure': False,
#     'retries': 1
# }

# # Define the DAG
# with DAG(
#     'gcs_file_pipeline_dag',
#     default_args=default_args,
#     description='Check GCS, stack CSVs into a pickle, and split the data',
#     schedule_interval=None,  # Triggered manually or on a schedule
# ) as dag:

#     # Task 1: Check for new files in GCS
#     check_for_files_task = PythonOperator(
#         task_id='check_for_new_files',
#         python_callable=check_for_new_files,
#         op_kwargs={
#             'bucket_name': 'airquality-mlops-rg',
#             'folder_path': 'api_data/'
#         },
#         provide_context=True,
#     )

#     # Task 2: Stack CSVs from GCS and save as a pickle
#     stack_csvs_task = PythonOperator(
#         task_id='stack_csvs',
#         python_callable=stack_csvs_to_pickle,
#         op_kwargs={
#             'bucket_name': 'airquality-mlops-rg',
#             'folder_path': 'api_data/',
#             'output_pickle_file': 'processed_data/stacked_air_pollution.pkl'
#         },
#     )
#     # Task to check bias
#     data_bias_task = PythonOperator(
#     task_id='perform_analysis',
#     python_callable=perform_analysis,
#     op_kwargs={
#         'bucket_name': 'airquality-mlops-rg',  # Replace with your GCS bucket name
#         'input_pickle_file': 'processed_data/stacked_air_pollution.pkl',
#         'sensitive_feature': 'month',
#         'threshold': 0.2,
#         'output_pickle_file': 'processed_data/resampled_data.pkl',
#     },
    
# )
#     # Task 3: Split the data
#     split_data_task = PythonOperator(
#         task_id='split_data',
#         python_callable=split_data,
#         op_kwargs={
#             'bucket_name': 'airquality-mlops-rg',
#             'input_pickle_file': 'processed_data/stacked_air_pollution.pkl',
#             'output_train_file': 'processed_data/train_data.pkl',
#             'output_test_file': 'processed_data/test_data.pkl'
#         },
#     )

#     schema_check_task = PythonOperator(
#     task_id='check_schema_and_generate_stats',
#     python_callable=generate_schema_and_statistics,
    
# )

#         # Dummy task to pull train file path
#     pull_train_file_task = PythonOperator(
#         task_id='pull_train_file',
#         python_callable=pull_train_file,
#         provide_context=True,  # Ensure context is provided
#     )

#         # Task 4: Pivot data
#     pivot_task = PythonOperator(
#         task_id='pivot_gcs_data',
#         python_callable=pivot_data_task,
#         op_kwargs={
#             'bucket_name': 'airquality-mlops-rg',
#             'input_file_path': "processed_data/train_data.pkl",  # Pull train file path from the previous task
#             'output_file_path': 'processed_data/pivoted_train_data.pkl',
#         },
#         provide_context=True,
#     )


#     clean_data_task = PythonOperator(
#         task_id='clean_data',
#         python_callable=clean_data,
#         op_kwargs={
#             'bucket_name': 'airquality-mlops-rg',
#             'input_file_path': 'processed_data/schema_gen.pkl',
#             'output_file_path': 'processed_data/cleaned_train_data.pkl',
#         }
#     )
    
#     # Define the task using PythonOperator
#     process_data_task = PythonOperator(
#         task_id='process_data',
#         python_callable=process_data,
#         op_kwargs={
#             'bucket_name': 'airquality-mlops-rg',
#             'input_file_path': 'processed_data/cleaned_train_data.pkl',
#             'output_file_path': 'processed_data/fill_na.pkl'
#         }
#     )

#  # Define the task using PythonOperator
#     check_anamoly_task = PythonOperator(
#         task_id='anomaly_detection_and_cleaning',
#         python_callable=anomaly_detection,
#         op_kwargs={
#             'bucket_name': 'airquality-mlops-rg',
#             'input_file_path': 'processed_data/fill_na.pkl',
#             'output_file_path': 'processed_data/check_anamoly.pkl'
#         }
#     )

#     feature_engineering_task = PythonOperator(
#         task_id='feature_engineering',
#         python_callable= feature_engineering,
#         op_kwargs={
#             'bucket_name': 'airquality-mlops-rg',
#             'input_file_path': 'processed_data/check_anamoly.pkl',
#             'output_file_path': 'processed_data/feature_engineer.pkl'
#         }
#     )
#     output_schema_check_task = PythonOperator(
#     task_id='check_schema_and_generate_stats',
#     python_callable= generate_output_schema_and_statistics,
# )
#     data_validation_task = PythonOperator(
#         task_id='data_validation',
#         python_callable=data_validation,
#         op_kwargs={
#             'bucket_name': 'airquality-mlops-rg',
#             'file_path': 'processed_data/pivoted_train_data.pkl',
#             'output_file_path': 'processed_data/schema_gen.pkl'
#         }
#     )

# # Define task dependencies
# check_for_files_task >> stack_csvs_task >> data_bias_task >> split_data_task >> schema_check_task >> pull_train_file_task >> pivot_task >> data_validation_task >> clean_data_task >> process_data_task >> check_anamoly_task >> feature_engineering_task >> output_schema_check_task 












from airflow import DAG
import os
import sys
from airflow import configuration as conf
import logging
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
from airflow.operators.dummy import DummyOperator
from datetime import timedelta,datetime
from airflow.utils.dates import days_ago
from my_operators.data_air import download_data_function
from my_operators.data_loader import stack_csvs_to_pickle
from my_operators.data_split import split
from my_operators.pivoting_data import pivot_parameters as pivoting_data_train
from my_operators.removal_of_uneccesary_cols import remove_uneccesary_cols as removal_of_uneccesary_cols_train
from my_operators.anamoly_detection import anamoly_detection_val as anamoly_detection_train
from my_operators.check_missing_values import handle_missing_vals as check_missing_values_train
from my_operators.feature_eng import feature_engineering as feature_eng_train
from my_operators.pivoting_data import pivot_parameters as pivoting_data_test
from my_operators.removal_of_uneccesary_cols import remove_uneccesary_cols as removal_of_uneccesary_cols_test
from my_operators.anamoly_detection import anamoly_detection_val as anamoly_detection_test
from my_operators.check_missing_values import handle_missing_vals as check_missing_values_test
from my_operators.feature_eng import feature_engineering as feature_eng_test
from my_operators.check_schema_original_airpollution import  main_generate_schema_and_statistics as main_check_schema_original
from my_operators.test_schema.check_output_data_schema import main_generate_schema_and_statistics as main_test_schema
from my_operators.check_output_data_schema import main_generate_schema_and_statistics as main_train_schema
from my_operators.data_bias_check_final import bias_main as data_biasing

# def check_anomalies_and_send_email(**kwargs):
#     anomalies = kwargs['ti'].xcom_pull(task_ids='download_data_from_api')
#     logging.info(f"Anomalies detected: {anomalies}")
#     if anomalies:  # If anomalies are detected, trigger the email
#         logging.info("Branching to send_anomaly_alert_api")
#         return 'send_anomaly_alert_api'
#     logging.info("Branching to continue_pipeline")
#     return 'continue_pipeline'

def check_anomalies_loading_data(**kwargs):
    anomalies = kwargs['ti'].xcom_pull(task_ids='load_data_pickle')
    logging.info(f"Anomalies detected: {anomalies}")
    if anomalies:  # If anomalies are detected, trigger the email
        logging.info("send_anomaly_alert_load_data")
        return 'send_anomaly_alert_load_data'
    logging.info("Branching to continue_pipeline")
    return 'continue_pipeline_load_data'

def check_anomalies_split_data(**kwargs):
    anomalies = kwargs['ti'].xcom_pull(task_ids='split_train_test')
    logging.info(f"Anomalies detected: {anomalies}")
    if anomalies:  # If anomalies are detected, trigger the email
        logging.info("send_anomaly_alert_train_test")
        return 'send_anomaly_alert_train_test'
    logging.info("Branching to continue_pipeline")
    return 'continue_pipeline_train_test'

def check_anomalies_pivoting_data_train(**kwargs):
    anomalies = kwargs['ti'].xcom_pull(task_ids='pivot_data_train')
    logging.info(f"Anomalies detected: {anomalies}")
    if anomalies:  # If anomalies are detected, trigger the email
        logging.info("send_anomaly_alert_pivot_data_train")
        return 'send_anomaly_alert_pivot_data_train'
    logging.info("Branching to continue_pipeline")
    return 'continue_pipeline_pivot_data_train'

def check_anomalies_pivoting_data_test(**kwargs):
    anomalies = kwargs['ti'].xcom_pull(task_ids='pivot_data_test')
    logging.info(f"Anomalies detected: {anomalies}")
    if anomalies:  # If anomalies are detected, trigger the email
        logging.info("send_anomaly_alert_pivot_data_test")
        return 'send_anomaly_alert_pivot_data_test'
    logging.info("Branching to continue_pipeline")
    return 'continue_pipeline_pivot_data_test'

def check_anomalies_removal_data_train(**kwargs):
    anomalies = kwargs['ti'].xcom_pull(task_ids='data_remove_cols_train')
    logging.info(f"Anomalies detected: {anomalies}")
    if anomalies:  # If anomalies are detected, trigger the email
        logging.info("send_anomaly_alert_remove_cols_train")
        return 'send_anomaly_alert_remove_cols_train'
    logging.info("Branching to continue_pipeline")
    return 'continue_pipeline_remove_cols_train'

def check_anomalies_removal_data_test(**kwargs):
    anomalies = kwargs['ti'].xcom_pull(task_ids='data_remove_cols_test')
    logging.info(f"Anomalies detected: {anomalies}")
    if anomalies:  # If anomalies are detected, trigger the email
        logging.info("send_anomaly_alert_remove_cols_test")
        return 'send_anomaly_alert_remove_cols_test'
    logging.info("Branching to continue_pipeline")
    return 'continue_pipeline_remove_cols_test'

def check_anomalies_missing_vals_train(**kwargs):
    anomalies = kwargs['ti'].xcom_pull(task_ids='handle_missing_vals_train')
    logging.info(f"Anomalies detected: {anomalies}")
    if anomalies:  # If anomalies are detected, trigger the email
        logging.info("send_anomaly_alert_remove_cols_train")
        return 'send_anomaly_alert_remove_cols_train'
    logging.info("Branching to continue_pipeline")
    return 'continue_pipeline_handle_missing_vals_train'

def check_anomalies_missing_vals_test(**kwargs):
    anomalies = kwargs['ti'].xcom_pull(task_ids='handle_missing_vals_test')
    logging.info(f"Anomalies detected: {anomalies}")
    if anomalies:  # If anomalies are detected, trigger the email
        logging.info("send_anomaly_alert_handle_missing_vals_test")
        return 'send_anomaly_alert_handle_missing_vals_test'
    logging.info("Branching to continue_pipeline")
    return 'continue_pipeline_handle_missing_vals_test'

def check_anomalies_anamoly_detection_val_train(**kwargs):
    anomalies = kwargs['ti'].xcom_pull(task_ids='anamolies_vals_train')
    logging.info(f"Anomalies detected: {anomalies}")
    if anomalies:  # If anomalies are detected, trigger the email
        logging.info("send_anomaly_alert_anamolies_vals_train")
        return 'send_anomaly_alert_anamolies_vals_train'
    logging.info("Branching to continue_pipeline")
    return 'continue_pipeline_anamolies_vals_train'

def check_anomalies_anamoly_detection_val_test(**kwargs):
    anomalies = kwargs['ti'].xcom_pull(task_ids='anamolies_vals_test')
    logging.info(f"Anomalies detected: {anomalies}")
    if anomalies:  # If anomalies are detected, trigger the email
        logging.info("send_anomaly_alert_anamolies_vals_test")
        return 'send_anomaly_alert_anamolies_vals_test'
    logging.info("Branching to continue_pipeline")
    return 'continue_pipeline_anamolies_vals_test'


default_args = {
    'owner': 'MLOPS',
    'start_date': datetime(2024, 10, 21),
    'retries': 1, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=1), # Delay before retries
}

dag = DAG(
    'datapipeline_new',
    default_args=default_args,
    description='Data Preprocessing pipeline using Airflow DAG',
    schedule_interval=None,
    catchup=False
)

branch_outliers_vals_train = BranchPythonOperator(
    task_id='check_anomalies_anamoly_detection_val_train',
    python_callable=check_anomalies_anamoly_detection_val_train,
    provide_context=True,
    dag=dag
)

branch_outliers_vals_test = BranchPythonOperator(
    task_id='check_anomalies_anamoly_detection_val_test',
    python_callable=check_anomalies_anamoly_detection_val_test,
    provide_context=True,
    dag=dag
)

branch_missing_vals_train = BranchPythonOperator(
    task_id='check_anomalies_missing_vals_train',
    python_callable=check_anomalies_missing_vals_train,
    provide_context=True,
    dag=dag
)

branch_missing_vals_test = BranchPythonOperator(
    task_id='check_anomalies_missing_vals_test',
    python_callable=check_anomalies_missing_vals_test,
    provide_context=True,
    dag=dag
)

branch_removal_data_train = BranchPythonOperator(
    task_id='check_anomalies_removal_data_train',
    python_callable=check_anomalies_removal_data_train,
    provide_context=True,
    dag=dag
)

branch_removal_data_test = BranchPythonOperator(
    task_id='check_anomalies_removal_data_test',
    python_callable=check_anomalies_removal_data_test,
    provide_context=True,
    dag=dag
)

# branch_task = BranchPythonOperator(
#     task_id='check_anomalies_and_send_email',
#     python_callable=check_anomalies_and_send_email,
#     provide_context=True,
#     dag=dag
# )

branch_task_load_data = BranchPythonOperator(
    task_id='check_anomalies_loading_data',
    python_callable=check_anomalies_loading_data,
    provide_context=True,
    dag=dag
)

branch_task_split = BranchPythonOperator(
    task_id='check_anomalies_split_data',
    python_callable=check_anomalies_split_data,
    provide_context=True,
    dag=dag
)

branch_pivot_data_train = BranchPythonOperator(
    task_id='check_anomalies_pivoting_data_train',
    python_callable=check_anomalies_pivoting_data_train,
    provide_context=True,
    dag=dag
)

branch_pivot_data_test = BranchPythonOperator(
    task_id='check_anomalies_pivoting_data_test',
    python_callable=check_anomalies_pivoting_data_test,
    provide_context=True,
    dag=dag
)

send_anomaly_alert_handle_missing_vals_test = EmailOperator(
    task_id='send_anomaly_alert_handle_missing_vals_test',
    to='anirudhak881@gmail.com',
    subject='Data Anomaly Alert for handling missing values in test',
    html_content="""<p>Anomalies detected in the data pipeline while handling missing values in test. Details:</p>
                    {% set anomalies = ti.xcom_pull(task_ids='handle_missing_vals_test') %}
                    {% if anomalies %}
                        <ul>{% for item in anomalies %}<li>{{ item }}</li>{% endfor %}</ul>
                    {% else %}
                        <p>No specific anomaly details available.</p>
                    {% endif %}""",
    dag=dag
)

send_anomaly_alert_handle_missing_vals_train = EmailOperator(
    task_id='send_anomaly_alert_handle_missing_vals_train',
    to='anirudhak881@gmail.com',
    subject='Data Anomaly Alert for handling missing values in train',
    html_content="""<p>Anomalies detected in the data pipeline while handling missing values in train. Details:</p>
                    {% set anomalies = ti.xcom_pull(task_ids='handle_missing_vals_train') %}
                    {% if anomalies %}
                        <ul>{% for item in anomalies %}<li>{{ item }}</li>{% endfor %}</ul>
                    {% else %}
                        <p>No specific anomaly details available.</p>
                    {% endif %}""",
    dag=dag
)

send_anomaly_removal_data_test = EmailOperator(
    task_id='send_anomaly_alert_remove_cols_test',
    to='anirudhak881@gmail.com',
    subject='Data Anomaly Alert for removing columns test',
    html_content="""<p>Anomalies detected in the data pipeline while removing columns test. Details:</p>
                    {% set anomalies = ti.xcom_pull(task_ids='data_remove_cols_test') %}
                    {% if anomalies %}
                        <ul>{% for item in anomalies %}<li>{{ item }}</li>{% endfor %}</ul>
                    {% else %}
                        <p>No specific anomaly details available.</p>
                    {% endif %}""",
    dag=dag
)

# removal cols anomalies
send_anomaly_removal_data_train = EmailOperator(
    task_id='send_anomaly_alert_remove_cols_train',
    to='anirudhak881@gmail.com',
    subject='Data Anomaly Alert for removing columns train',
    html_content="""<p>Anomalies detected in the data pipeline removing columns train. Details:</p>
                    {% set anomalies = ti.xcom_pull(task_ids='data_remove_cols_train') %}
                    {% if anomalies %}
                        <ul>{% for item in anomalies %}<li>{{ item }}</li>{% endfor %}</ul>
                    {% else %}
                        <p>No specific anomaly details available.</p>
                    {% endif %}""",
    dag=dag
)

send_anomaly_alert = EmailOperator(
    task_id='send_anomaly_alert_api',
    to='anirudhak881@gmail.com',
    subject='Data Anomaly Alert for API',
    html_content="""<p>Anomalies detected in the data pipeline while using API. Details:</p>
                    {% set anomalies = ti.xcom_pull(task_ids='download_data_from_api') %}
                    {% if anomalies %}
                        {% if anomalies is string %}
                            <ul><li>{{ anomalies }}</li></ul>
                        {% else %}
                            <ul>{% for item in anomalies %}<li>{{ item }}</li>{% endfor %}</ul>
                        {% endif %}
                    {% else %}
                        <p>No specific anomaly details available.</p>
                    {% endif %}""",
    dag=dag
)

send_anomaly_alert_load_data = EmailOperator(
    task_id='send_anomaly_alert_load_data',
    to='anirudhak881@gmail.com',
    subject='Data Anomaly Alert for Loading Data',
    html_content="""<p>Anomalies detected in the data pipeline while loading data. Details:</p>
                    {% set anomalies = ti.xcom_pull(task_ids='load_data_pickle') %}
                    {% if anomalies %}
                        <ul>{% for item in anomalies %}<li>{{ item }}</li>{% endfor %}</ul>
                    {% else %}
                        <p>No specific anomaly details available.</p>
                    {% endif %}""",
    dag=dag
)

send_anomaly_alert_train_test = EmailOperator(
    task_id='send_anomaly_alert_train_test',
    to='anirudhak881@gmail.com',
    subject='Data Anomaly Alert for Splitting train Data',
    html_content="""<p>Anomalies detected in the data pipeline while splitting data. Details:</p>
                    {% set anomalies = ti.xcom_pull(task_ids='split_train_test') %}
                    {% if anomalies %}
                        <ul>{% for item in anomalies %}<li>{{ item }}</li>{% endfor %}</ul>
                    {% else %}
                        <p>No specific anomaly details available.</p>
                    {% endif %}""",
    dag=dag
)

send_anomaly_pivot_data_train = EmailOperator(
    task_id='send_anomaly_pivot_data_train',
    to='anirudhak881@gmail.com',
    subject='Data Anomaly Alert for pivoting train Data',
    html_content="""<p>Anomalies detected in the data pipeline while pivotting data. Details:</p>
                    {% set anomalies = ti.xcom_pull(task_ids='pivot_data_train') %}
                    {% if anomalies %}
                        <ul>{% for item in anomalies %}<li>{{ item }}</li>{% endfor %}</ul>
                    {% else %}
                        <p>No specific anomaly details available.</p>
                    {% endif %}""",
    dag=dag
)

send_anomaly_alert_anamolies_vals_test= EmailOperator(
    task_id='send_anomaly_alert_anamolies_vals_test',
    to='anirudhak881@gmail.com',
    subject='Data Anomaly Alert for detecting outliers and negative values test data',
    html_content="""<p> detecting outliers and negative values. Details:</p>
                    {% set anomalies = ti.xcom_pull(task_ids='anamolies_vals_test') %}
                    {% if anomalies %}
                        <ul>{% for item in anomalies %}<li>{{ item }}</li>{% endfor %}</ul>
                    {% else %}
                        <p>No specific anomaly details available.</p>
                    {% endif %}""",
    dag=dag
)

send_anomaly_alert_anamolies_vals_train= EmailOperator(
    task_id='send_anomaly_alert_anamolies_vals_train',
    to='anirudhak881@gmail.com',
    subject='Data Anomaly Alert for detecting outliers and negative values training data',
    html_content="""<p> detecting outliers and negative values. Details:</p>
                    {% set anomalies = ti.xcom_pull(task_ids='anamolies_vals_train') %}
                    {% if anomalies %}
                        <ul>{% for item in anomalies %}<li>{{ item }}</li>{% endfor %}</ul>
                    {% else %}
                        <p>No specific anomaly details available.</p>
                    {% endif %}""",
    dag=dag
)

send_anomaly_pivot_data_test = EmailOperator(
    task_id='send_anomaly_pivot_data_test',
    to='anirudhak881@gmail.com',
    subject='Data Anomaly Alert for pivoting test Data',
    html_content="""<p>Anomalies detected in the data pipeline while pivotting data. Details:</p>
                    {% set anomalies = ti.xcom_pull(task_ids='pivot_data_test') %}
                    {% if anomalies %}
                        <ul>{% for item in anomalies %}<li>{{ item }}</li>{% endfor %}</ul>
                    {% else %}
                        <p>No specific anomaly details available.</p>
                    {% endif %}""",
    dag=dag
)

#continue_pipeline = DummyOperator(task_id='continue_pipeline', dag=dag)

continue_pipeline_load_data = DummyOperator(task_id='continue_pipeline_load_data', dag=dag)

continue_pipeline_train_test = DummyOperator(task_id='continue_pipeline_train_test',dag=dag)

continue_pipeline_pivot_data_train = DummyOperator(task_id='continue_pipeline_pivot_data_train',dag=dag)

continue_pipeline_pivot_data_test = DummyOperator(task_id='continue_pipeline_pivot_data_test',dag=dag)

continue_pipeline_remove_cols_train = DummyOperator(task_id='continue_pipeline_remove_cols_train',dag=dag)

continue_pipeline_remove_cols_test = DummyOperator(task_id='continue_pipeline_remove_cols_test',dag=dag)

continue_pipeline_handle_missing_vals_test = DummyOperator(task_id='continue_pipeline_handle_missing_vals_test',dag=dag)

continue_pipeline_handle_missing_vals_train = DummyOperator(task_id='continue_pipeline_handle_missing_vals_train',dag=dag)

continue_pipeline_anamolies_vals_test = DummyOperator(task_id='continue_pipeline_anamolies_vals_test',dag=dag)

continue_pipeline_anamolies_vals_train = DummyOperator(task_id='continue_pipeline_anamolies_vals_train',dag=dag)

#merge_branch = DummyOperator(task_id='merge_branch', trigger_rule='none_failed_min_one_success',dag=dag)

merge_branch_load_data = DummyOperator(task_id='merge_branch_load_data', trigger_rule='none_failed_min_one_success',dag=dag)

merge_branch_train_test = DummyOperator(task_id='merge_branch_train_test', trigger_rule='none_failed_min_one_success',dag=dag)

merge_branch_pivot_data_train= DummyOperator(task_id='merge_branch_pivot_data_train', trigger_rule='none_failed_min_one_success',dag=dag)

merge_branch_pivot_data_test= DummyOperator(task_id='merge_branch_pivot_data_test', trigger_rule='none_failed_min_one_success',dag=dag)

merge_branch_remove_cols_train= DummyOperator(task_id='merge_branch_remove_cols_train', trigger_rule='none_failed_min_one_success',dag=dag)

merge_branch_remove_cols_test= DummyOperator(task_id='merge_branch_remove_cols_test', trigger_rule='none_failed_min_one_success',dag=dag)

merge_branch_handle_missing_vals_train= DummyOperator(task_id='merge_branch_handle_missing_vals_train', trigger_rule='none_failed_min_one_success',dag=dag)

merge_branch_handle_missing_vals_test= DummyOperator(task_id='merge_branch_handle_missing_vals_test', trigger_rule='none_failed_min_one_success',dag=dag)

merge_branch_anamoly_detection_val_train= DummyOperator(task_id='merge_branch_anamoly_detection_val_train', trigger_rule='none_failed_min_one_success',dag=dag)

merge_branch_anamoly_detection_val_test= DummyOperator(task_id='merge_branch_anamoly_detection_val_test', trigger_rule='none_failed_min_one_success',dag=dag)

# download the data in form of csv using data api 
# download_data_api = PythonOperator(
#     task_id='download_data_from_api',
#     python_callable=download_data_function,
#     dag=dag
# )

# load the data and save it in pickle file
data_Loader = PythonOperator(
    task_id='load_data_pickle',
    python_callable=stack_csvs_to_pickle,
    dag=dag
)

with DAG('gcs_csv_stack_dag', default_args=default_args, schedule_interval=None) as dag:

    stack_csvs_task = PythonOperator(
        task_id='stack_csvs',
        python_callable=stack_csvs_to_pickle,
        op_kwargs={
            'bucket_name': 'airquality-mlops-rg',
            'folder_path': 'air_pollution_raw/',
            'output_pickle_file': 'processed_data/air_pollution.pkl'
        },
    )

data_Bias = PythonOperator(
    task_id='bias_detection_and_mitigation',
    python_callable=data_biasing,
    dag=dag)

# split data into traning and testing
data_Split = PythonOperator(
    task_id='split_train_test',
    python_callable=split,
    dag=dag
)

data_schema_original = PythonOperator(
    task_id = 'check_schema_of_original_air_data',
    python_callable = main_check_schema_original,
    dag = dag
)

# pivot the data to contain pm2.5 parameters for train data
data_train_pivot = PythonOperator(
    task_id='pivot_data_train',
    python_callable=pivoting_data_train,
    dag=dag
)

# pivot the data to contain pm2.5 parameters for test data
data_test_pivot = PythonOperator(
    task_id='pivot_data_test',
    python_callable=pivoting_data_test,
    dag=dag
)

# remove values regarding other gases train data
data_remove_cols_train = PythonOperator(
    task_id='data_remove_cols_train',
    python_callable=removal_of_uneccesary_cols_train,
    dag=dag
)

# remove values regarding other gases test data
data_remove_cols_test = PythonOperator(
    task_id='data_remove_cols_test',
    python_callable=removal_of_uneccesary_cols_test,
    dag=dag
)

# handle missing values train data
handle_missing_vals_train = PythonOperator(
    task_id='handle_missing_vals_train',
    python_callable=check_missing_values_train,
    dag=dag
)

# handle missing values test data
handle_missing_vals_test = PythonOperator(
    task_id='handle_missing_vals_test',
    python_callable=check_missing_values_test,
    dag=dag
)

# handle anamolies for train data
anamolies_vals_train = PythonOperator(
    task_id='anamolies_vals_train',
    python_callable=anamoly_detection_train,
    dag=dag
)

#handle anamolies for test data
anamolies_vals_test = PythonOperator(
    task_id='anamolies_vals_test',
    python_callable=anamoly_detection_test,
    dag=dag
)

# feature engineering for train data
feature_engineering_train = PythonOperator(
    task_id='feature_engineering_train',
    python_callable=feature_eng_train,
    dag=dag
)

data_schema_train_data_feature_eng = PythonOperator(
    task_id = 'schema_train_data_final_processed',
    python_callable = main_train_schema,
    dag = dag
)

data_schema_test_data_feature_eng = PythonOperator(
    task_id = 'schema_test_data_final_processed',
    python_callable = main_test_schema,
    dag = dag
)

# feature engineering for test data
feature_engineering_test = PythonOperator(
    task_id='feature_engineering_test',
    python_callable=feature_eng_test,
    dag=dag
)

# order in which tasks are run
#download_data_api >> branch_task >> [send_anomaly_alert, continue_pipeline] >> merge_branch \
data_Loader >> branch_task_load_data >> [send_anomaly_alert_load_data,continue_pipeline_load_data] >> merge_branch_load_data \
>> data_Bias \
>> data_Split >> branch_task_split >> [send_anomaly_alert_train_test,continue_pipeline_train_test] >> merge_branch_train_test \
>> data_schema_original \
>> data_train_pivot >> branch_pivot_data_train >> [send_anomaly_pivot_data_train,continue_pipeline_pivot_data_train] \
>> merge_branch_pivot_data_train \
>> data_remove_cols_train >> branch_removal_data_train \
>> [send_anomaly_removal_data_train,continue_pipeline_remove_cols_train] >> merge_branch_remove_cols_train \
>> handle_missing_vals_train >> branch_missing_vals_train \
>> [send_anomaly_alert_handle_missing_vals_train,continue_pipeline_handle_missing_vals_train] \
>> merge_branch_handle_missing_vals_train \
>> anamolies_vals_train >> branch_outliers_vals_train >> [send_anomaly_alert_anamolies_vals_train,continue_pipeline_anamolies_vals_train] \
>> merge_branch_anamoly_detection_val_train \
>> feature_engineering_train \
>> data_test_pivot >> branch_pivot_data_test >> [send_anomaly_pivot_data_test,continue_pipeline_pivot_data_test] \
>> merge_branch_pivot_data_test \
>> data_remove_cols_test >> branch_removal_data_test >> [send_anomaly_removal_data_test,continue_pipeline_remove_cols_test] \
>> merge_branch_remove_cols_test \
>> handle_missing_vals_test >> branch_missing_vals_test \
>> [send_anomaly_alert_handle_missing_vals_test,continue_pipeline_handle_missing_vals_test] \
>> merge_branch_handle_missing_vals_test \
>> anamolies_vals_test >> branch_outliers_vals_test >> [send_anomaly_alert_anamolies_vals_test,continue_pipeline_anamolies_vals_test] \
>> merge_branch_anamoly_detection_val_test \
>> feature_engineering_test \
>> data_schema_train_data_feature_eng \
>> data_schema_test_data_feature_eng

if __name__ == "__main__":
    dag.cli()



