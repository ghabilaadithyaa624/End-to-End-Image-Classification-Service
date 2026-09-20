variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-south-1"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "image-classification-mlops"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "dev"
}

variable "kubernetes_version" {
  description = "EKS Kubernetes version"
  type        = string
  default     = "1.33"
}

variable "alert_email" {
  description = "Email address for AWS billing budget alerts"
  type        = string
  default     = "ghabilaadithyaa624@gmail.com"
}

variable "budget_limit_amount" {
  description = "Monthly budget limit in USD to protect AWS credits"
  type        = string
  default     = "50"
}
