# Terraform Versions and Provider Requirements (Phase 5 DevOps IaC)
terraform {
  required_version = ">= 1.5.0"

  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0.2"
    }
  }
}

provider "docker" {
  # Connects to default local Docker daemon socket
  # Windows named pipe: npipe:////./pipe/docker_engine
  # Linux/Unix socket: unix:///var/run/docker.sock
}
