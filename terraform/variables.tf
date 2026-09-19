variable "environment" {
  description = "Deployment environment (e.g. dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "region" {
  description = "Cloud region for resources"
  type        = string
  default     = "us-central1"
}

variable "project_name" {
  description = "Project name identifier"
  type        = string
  default     = "image-classification-mlops"
}

variable "app_replicas" {
  description = "Number of application pod replicas"
  type        = number
  default     = 2
}
