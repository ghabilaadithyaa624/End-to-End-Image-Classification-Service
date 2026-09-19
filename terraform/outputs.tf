output "service_name" {
  description = "The deployed service name identifier"
  value       = local.service_name
}

output "environment" {
  description = "Target deployment environment"
  value       = var.environment
}

output "region" {
  description = "Target cloud region"
  value       = var.region
}
