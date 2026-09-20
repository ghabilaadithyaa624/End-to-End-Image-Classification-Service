output "cluster_name" {
  description = "EKS Cluster Name"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS Cluster API Endpoint"
  value       = module.eks.cluster_endpoint
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "ecr_repository_url" {
  description = "Amazon ECR Repository URL for Docker images"
  value       = aws_ecr_repository.app.repository_url
}

output "s3_bucket_name" {
  description = "Amazon S3 Bucket Name for MLflow Model Artifacts"
  value       = aws_s3_bucket.mlflow_artifacts.id
}

output "mlflow_artifacts_bucket" {
  description = "MLflow artifact bucket alias"
  value       = aws_s3_bucket.mlflow_artifacts.bucket
}
