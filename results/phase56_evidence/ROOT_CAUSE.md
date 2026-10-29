# Root Cause Analysis: Grafana Dashboard Visualization Failure

## Finding: **C) both**

Both **Cause A** (panel datasource UID mismatch) and **Cause B** (datasource URL unreachable from inside the Grafana container) are present simultaneously and caused every panel on dashboard `xai-ops-main` to enter an error state (`No data` with red error triangles).

---

### Cause A: Panel Datasource UID Mismatch
1. **Provisioned Datasource State**:
   In `monitoring/grafana/provisioning/datasources/prometheus.yml`, no explicit `uid` field was defined:
   ```yaml
   apiVersion: 1
   datasources:
     - name: Prometheus
       type: prometheus
       access: proxy
       url: http://prometheus:9090
       isDefault: true
   ```
   Because Grafana did not find a specified `uid`, it auto-generated a random hash UID at initialization:
   ```json
   {
     "id": 1,
     "uid": "PBFA97CFB590B2093",
     "name": "Prometheus",
     "type": "prometheus",
     "url": "http://prometheus:9090",
     "isDefault": true
   }
   ```
2. **Dashboard Panel Configuration**:
   In `monitoring/grafana/dashboards/xai-operations.json`, all 8 panels specified:
   ```json
   "datasource": { "type": "prometheus", "uid": "Prometheus" }
   ```
   Grafana's query broker could not locate any datasource with `uid: "Prometheus"` because the active datasource UID was `"PBFA97CFB590B2093"`.

---

### Cause B: Datasource URL Unreachable from Inside Grafana Container
1. **Configured URL**:
   The datasource provisioning file configured `url: http://prometheus:9090`.
2. **Docker Network & Resolution**:
   Both `xai-grafana-tf` and `xai-prometheus-tf` run on bridge network `xai_monitoring` created by Terraform (`terraform/main.tf`).
   In Terraform, the container resource is named `xai-prometheus-tf` with no network alias for `prometheus`.
3. **Execution Output from inside `xai-grafana-tf`**:
   - Command: `docker exec xai-grafana-tf wget -qO- 'http://prometheus:9090/api/v1/query?query=up'`
     **Output**: `wget: bad address 'prometheus:9090'` (**FAILED**)
   - Command: `docker exec xai-grafana-tf wget -qO- 'http://xai-prometheus-tf:9090/api/v1/query?query=up'`
     **Output**: `{"status":"success","data":{"resultType":"vector","result":[...]}}` (**SUCCEEDED**)
   - Command: `docker exec xai-grafana-tf wget -qO- 'http://host.docker.internal:9090/api/v1/query?query=up'`
     **Output**: `{"status":"success","data":{"resultType":"vector","result":[...]}}` (**SUCCEEDED**)

---

### Conclusion
Even if the datasource UID were matched, queries would fail with DNS resolution error `bad address 'prometheus:9090'`. Even if the URL were reachable, queries would fail due to the missing/mismatched datasource UID. Hence, **C) both** is the exact root cause.
