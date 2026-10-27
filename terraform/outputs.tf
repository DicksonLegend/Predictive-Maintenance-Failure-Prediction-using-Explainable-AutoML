# Terraform Outputs for Monitoring Stack

output "prometheus_url" {
  description = "Direct browser access URL for Prometheus time-series metric server"
  value       = "http://localhost:${var.prometheus_port}"
}

output "grafana_url" {
  description = "Direct browser access URL for Grafana operational dashboard"
  value       = "http://localhost:${var.grafana_port}"
}

output "network_name" {
  description = "Name of the provisioned Docker bridge network"
  value       = docker_network.xai_monitoring.name
}

output "prometheus_container_id" {
  description = "Docker container ID for Prometheus"
  value       = docker_container.prometheus.id
}

output "grafana_container_id" {
  description = "Docker container ID for Grafana"
  value       = docker_container.grafana.id
}
