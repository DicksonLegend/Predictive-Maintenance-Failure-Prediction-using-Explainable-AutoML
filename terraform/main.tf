# Terraform Infrastructure as Code: Monitoring Stack (Prometheus & Grafana)

# 1. Dedicated Docker Bridge Network
resource "docker_network" "xai_monitoring" {
  name   = "xai_monitoring"
  driver = "bridge"
}

# 2. Persistent Volume for Grafana User Data and Session State
resource "docker_volume" "grafana_data" {
  name = "xai_grafana_data"
}

# 3. Container Images
resource "docker_image" "prometheus" {
  name         = "prom/prometheus:latest"
  keep_locally = true
}

resource "docker_image" "grafana" {
  name         = "grafana/grafana:latest"
  keep_locally = true
}

# 4. Prometheus Time-Series Metrics Server Container
resource "docker_container" "prometheus" {
  name    = "xai-prometheus-tf"
  image   = docker_image.prometheus.image_id
  restart = "unless-stopped"

  networks_advanced {
    name = docker_network.xai_monitoring.name
  }

  ports {
    internal = 9090
    external = var.prometheus_port
  }

  host {
    host = var.api_host
    ip   = "host-gateway"
  }

  command = [
    "--config.file=/etc/prometheus/prometheus.yml",
    "--storage.tsdb.path=/prometheus",
    "--web.console.libraries=/usr/share/prometheus/console_libraries",
    "--web.console.templates=/usr/share/prometheus/consoles",
    "--web.enable-lifecycle"
  ]

  volumes {
    host_path      = abspath("${path.module}/../monitoring/prometheus/prometheus.yml")
    container_path = "/etc/prometheus/prometheus.yml"
    read_only      = true
  }

  volumes {
    host_path      = abspath("${path.module}/../monitoring/prometheus/alerts.yml")
    container_path = "/etc/prometheus/alerts.yml"
    read_only      = true
  }

  healthcheck {
    test     = ["CMD", "wget", "-q", "--tries=1", "--spider", "http://localhost:9090/-/healthy"]
    interval = "15s"
    timeout  = "5s"
    retries  = 3
  }
}

# 5. Grafana Analytics & Operational Dashboard Container
resource "docker_container" "grafana" {
  name    = "xai-grafana-tf"
  image   = docker_image.grafana.image_id
  restart = "unless-stopped"

  depends_on = [
    docker_container.prometheus
  ]

  networks_advanced {
    name = docker_network.xai_monitoring.name
  }

  ports {
    internal = 3000
    external = var.grafana_port
  }

  env = [
    "GF_SECURITY_ADMIN_USER=admin",
    "GF_SECURITY_ADMIN_PASSWORD=${var.grafana_admin_password}",
    "GF_USERS_ALLOW_SIGN_UP=false"
  ]

  volumes {
    volume_name    = docker_volume.grafana_data.name
    container_path = "/var/lib/grafana"
  }

  volumes {
    host_path      = abspath("${path.module}/../monitoring/grafana/provisioning/datasources")
    container_path = "/etc/grafana/provisioning/datasources"
    read_only      = true
  }

  volumes {
    host_path      = abspath("${path.module}/../monitoring/grafana/provisioning/dashboards")
    container_path = "/etc/grafana/provisioning/dashboards"
    read_only      = true
  }

  volumes {
    host_path      = abspath("${path.module}/../monitoring/grafana/dashboards")
    container_path = "/etc/grafana/dashboards"
    read_only      = true
  }

  healthcheck {
    test     = ["CMD", "wget", "-q", "--tries=1", "--spider", "http://localhost:3000/api/health"]
    interval = "15s"
    timeout  = "5s"
    retries  = 3
  }
}
