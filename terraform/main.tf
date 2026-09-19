terraform {
  required_version = ">= 1.5.0"
  required_providers {
    local = {
      source  = "hashicorp/local"
      version = "~> 2.4.0"
    }
  }
}

locals {
  service_name = "${var.project_name}-${var.environment}"
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# Example local resource representing infrastructure state
resource "local_file" "deployment_manifest" {
  filename = "${path.module}/generated_spec.json"
  content = jsonencode({
    service_name = local.service_name
    environment  = var.environment
    region       = var.region
    replicas     = var.app_replicas
    tags         = local.common_tags
  })
}
