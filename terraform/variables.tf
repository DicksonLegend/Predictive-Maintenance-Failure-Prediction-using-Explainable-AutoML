# Terraform Input Variables for Monitoring Infrastructure

variable "grafana_admin_password" {
  description = "Administrator password for Grafana web dashboard login"
  type        = string
  default     = "admin"
  sensitive   = true

  validation {
    condition     = length(var.grafana_admin_password) >= 5
    error_message = "The Grafana admin password must be at least 5 characters in length."
  }
}

variable "prometheus_port" {
  description = "Host port exposed for Prometheus time-series database web interface"
  type        = number
  default     = 9090
}

variable "grafana_port" {
  description = "Host port exposed for Grafana operations dashboard"
  type        = number
  default     = 3000
}

variable "api_host" {
  description = "Host IP address or DNS mapping for the FastAPI application telemetry scrape endpoint"
  type        = string
  default     = "host.docker.internal"
}
