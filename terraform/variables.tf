# variables.tf

variable "project_id" {
  description = "The ID of the project where resources will be created"
  type        = string
}

variable "region" {
  description = "Region where resources will be created"
  type        = string
  default     = "us-central1"
}

variable "data_bucket_name" {
  description = "Name of the GCS bucket for data storage"
  type        = string
}

variable "composer_env_name" {
  description = "Name of the Cloud Composer environment"
  type        = string
}

