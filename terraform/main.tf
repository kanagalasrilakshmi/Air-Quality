# Provider block for Google Cloud
provider "google" {
  credentials = file("/Users/swapnavippaturi/Downloads/key.json")
  project     = var.project_id  # GCP Project ID
  region      = var.region      # GCP Region
}

# Variables for project and environment configuration
variable "bucket_name" {
  description = "The name of the Cloud Storage bucket for Composer DAGs"
  type        = string
}

variable "serviceAccount" {
  description = "The service account email"
  type        = string
}

# Create a random ID for uniqueness in bucket name
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# Create a Cloud Storage bucket for Composer DAGs
resource "google_storage_bucket" "airbucketname" {
  name           = "${var.bucket_name}-${random_id.bucket_suffix.hex}"  # Ensure globally unique name
  location       = var.region
  force_destroy  = true  # Automatically delete bucket contents when destroying
}

# Create IAM service account for Composer
resource "google_service_account" "composer_service_account" {
  account_id   = "composer-service-account"
  display_name = "Composer Service Account"
}

# Grant necessary roles to the Composer service account
resource "google_project_iam_member" "composer_service_account_roles" {
  for_each = toset([
    "roles/composer.admin",      # Admin for Cloud Composer
    "roles/storage.admin",       # Admin for Cloud Storage
    "roles/iam.serviceAccountUser",  # IAM Service Account User role
    "roles/logging.logWriter",   # Logs access
    "roles/monitoring.viewer"    # Monitoring access
  ])

  project = var.project_id
  member  = "serviceAccount:${google_service_account.composer_service_account.email}"
  role    = each.value
}

resource "google_composer_environment" "airquality-composer" {
  name   = "airquality-composer"
  region = "us-central1"

  config {
    software_config {
      image_version = "composer-2.9.7-airflow-2.9.3"  # Use a valid image version
      pypi_packages = {
                           "gs://airquality-mlops-rg/composer_requirements/requirements.txt" = ""
      } 
      env_variables = {
        LOAD_BUCKET_NAME                 = "airquality-mlops-rg"
        FOLDER_PATH                      = "api_data/"
        LOAD_OUTPUT_PICKLE_FILE          = "processed/air_pollution.pkl"
        BIAS_CHECK_INPUT_PATH            = "processed/air_pollution.pkl"
        BIAS_CHECK_OUTPUT_PATH           = "processed/resampled_data.pkl"
        BIAS_CHECK_SENSITIVE_FEATURE     = "month"
        BIAS_CHECK_THRESHOLD             = "0.2"
        DATA_SPLIT_BUCKET_NAME           = "airquality-mlops-rg"
        DATA_SPLIT_INPUT_FILE_PATH       = "processed/resampled_data.pkl"
        DATA_SPLIT_TRAIN_OUTPUT_FILE_PATH= "processed/train/train_data.pkl"
        DATA_SPLIT_TEST_OUTPUT_FILE_PATH = "processed/test/test_data.pkl"
        SCHEMA_STATS_BUCKET_NAME         = "airquality-mlops-rg"
        SCHEMA_STATS_INPUT_FILE_PATH     = "processed/resampled_data.pkl"
        SCHEMA_FILE_OUTPUT_PATH          = "processed/Schema.pkl"
        STATS_FILE_OUTPUT_PATH           = "processed/Stats.pkl"
        DATA_BUCKET_NAME                 = "airquality-mlops-rg"
        TRAIN_DATA_PIVOT_INPUT           = "processed/train/train_data.pkl"
        TRAIN_DATA_PIVOT_OUTPUT          = "processed/train/pivot_data.pkl"
        TRAIN_DATA_RM_COL_INPUT          = "processed/train/pivot_data.pkl"
        TRAIN_DATA_RM_COL_OUTPUT         = "processed/train/remove_col_data.pkl"
        TRAIN_DATA_MISS_VAL_INPUT        = "processed/train/remove_col_data.pkl"
        TRAIN_DATA_MISS_VAL_OUTPUT       = "processed/train/missing_val_data.pkl"
        TRAIN_DATA_ANMLY_INPUT          = "processed/train/missing_val_data.pkl"
        TRAIN_DATA_ANMLY_OUTPUT         = "processed/train/anamoly_data.pkl"
        TRAIN_DATA_FEA_ENG_INPUT         = "processed/train/missing_val_data.pkl"
        TRAIN_DATA_FEA_ENG_OUTPUT        = "processed/train/feature_eng_data.pkl"
        TRAIN_SCHEMA_FILE_PATH           = "processed/train/output_schema.pkl"
        TRAIN_STATS_FILE_PATH            = "processed/train/output_stats.pkl"
        TRAIN_DATASET_FILE_PATH          = "processed/train/feature_eng_data.pkl"
        TEST_DATA_PIVOT_INPUT            = "processed/test/train_data.pkl"
        TEST_DATA_PIVOT_OUTPUT           = "processed/test/pivot_data.pkl"
        TEST_DATA_RM_COL_INPUT           = "processed/test/pivot_data.pkl"
        TEST_DATA_RM_COL_OUTPUT          = "processed/test/remove_col_data.pkl"
        TEST_DATA_MISS_VAL_INPUT         = "processed/test/remove_col_data.pkl"
        TEST_DATA_MISS_VAL_OUTPUT        = "processed/test/missing_val_data.pkl"
        TEST_DATA_ANMLY_INPUT           = "processed/test/missing_val_data.pkl"
        TEST_DATA_ANMLY_OUTPUT          = "processed/test/anamoly_data.pkl"
        TEST_DATA_FEA_ENG_INPUT          = "processed/test/missing_val_data.pkl"
        TEST_DATA_FEA_ENG_OUTPUT         = "processed/test/feature_eng_data.pkl"
        TEST_SCHEMA_FILE_PATH           = "processed/train/output_schema.pkl"
        TEST_STATS_FILE_PATH            = "processed/train/output_stats.pkl"
        TEST_DATASET_FILE_PATH          = "processed/train/feature_eng_data.pkl"
       #AIRFLOW__SMTP__SMTP_HOST        = "smtp.gmail.com"
       # AIRFLOW__SMTP__SMTP_TLS         = "True"
        #AIRFLOW__SMTP__SMTP_STARTTLS    = "True"
       # AIRFLOW__SMTP__SMTP_SSL        = "False"
        #AIRFLOW__SMTP__SMTP_USER       = "anirudhak881@gmail.com"
       # AIRFLOW__SMTP__SMTP_PASSWORD   = "ksmhwqrzhjllqsaz"  # Remember to replace with your actual app password
       # AIRFLOW__SMTP__SMTP_PORT       = "587"
       # AIRFLOW__SMTP__SMTP_MAIL_FROM  = "anirudhak881@gmail.com"       
        #AIRFLOW__SMTP__SMTP_DEFAULT_SENDER = "anirudhak881@gmail.com"
        #AIRFLOW__SMTP__SMTP_DEFAULT_RECIPIENT = "default-recipient@example.com" 
      }
    }
       
  }
}
# Optional: Set up the network and subnetwork (if required)
resource "google_compute_network" "composer_network" {
  name                    = "composer-network"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "composer_subnetwork" {
  name          = "composer-subnetwork"
  network       = google_compute_network.composer_network.name
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
}
