"""Synthetic Load Generator for Predictive Maintenance Inference & Metrics Validation.

Sends randomized realistic telemetry payloads across nominal (low risk), degraded (medium risk),
and critical (high risk) operating profiles to populate Prometheus and Grafana dashboards.
"""
import argparse
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import httpx

# Telemetry profiles reflecting low, medium, and high failure probabilities
TELEMETRY_PROFILES = [
    # Nominal Low Risk profile (air: 298-300K, speed: 1500-1600rpm, torque: 35-42Nm, wear: 10-60min)
    {
        "type": "L",
        "air_temp": (298.0, 300.5),
        "process_temp": (308.0, 310.5),
        "rotational_speed": (1500.0, 1600.0),
        "torque": (35.0, 42.0),
        "tool_wear": (10.0, 60.0),
    },
    # Moderate / Degraded Medium Risk profile (elevated temp & wear)
    {
        "type": "M",
        "air_temp": (301.0, 303.0),
        "process_temp": (311.0, 313.0),
        "rotational_speed": (1380.0, 1450.0),
        "torque": (48.0, 58.0),
        "tool_wear": (140.0, 190.0),
    },
    # Critical High Risk profile (severe torque overload or extreme tool wear)
    {
        "type": "H",
        "air_temp": (303.5, 305.0),
        "process_temp": (312.5, 314.5),
        "rotational_speed": (1250.0, 1340.0),
        "torque": (62.0, 75.0),
        "tool_wear": (205.0, 245.0),
    },
]


def generate_payload():
    """Generates a realistic telemetry dictionary by sampling from operating profiles."""
    profile = random.choices(TELEMETRY_PROFILES, weights=[0.60, 0.25, 0.15])[0]
    return {
        "type": profile["type"],
        "air_temp": round(random.uniform(*profile["air_temp"]), 1),
        "process_temp": round(random.uniform(*profile["process_temp"]), 1),
        "rotational_speed": round(random.uniform(*profile["rotational_speed"]), 0),
        "torque": round(random.uniform(*profile["torque"]), 1),
        "tool_wear": round(random.uniform(*profile["tool_wear"]), 0),
    }


def send_request(client: httpx.Client, endpoint: str):
    """Sends a single inference request and returns the HTTP status code and latency."""
    payload = generate_payload()
    start = time.perf_counter()
    try:
        response = client.post(endpoint, json=payload, timeout=5.0)
        latency = time.perf_counter() - start
        return response.status_code, latency
    except Exception:
        return 0, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description="Synthetic Load Generator for XAI API")
    parser.add_argument("--requests", type=int, default=300, help="Total number of requests to fire")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent worker threads")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Base URL of target API")
    args = parser.parse_args()

    endpoint = f"{args.url.rstrip('/')}/predict"
    print("=" * 65)
    print("XAI PREDICTIVE MAINTENANCE — SYNTHETIC LOAD GENERATOR")
    print("=" * 65)
    print(f"Target URL:    {endpoint}")
    print(f"Total Requests: {args.requests}")
    print(f"Concurrency:   {args.concurrency}")
    print("Firing requests...")

    status_counts = Counter()
    latencies = []

    start_wall = time.perf_counter()
    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(send_request, client, endpoint) for _ in range(args.requests)]
            for fut in as_completed(futures):
                status, lat = fut.result()
                status_counts[status] += 1
                latencies.append(lat)

    total_time = time.perf_counter() - start_wall
    achieved_rps = args.requests / total_time if total_time > 0 else 0
    avg_latency_ms = (sum(latencies) / len(latencies) * 1000) if latencies else 0

    print("-" * 65)
    print("LOAD GENERATION COMPLETE!")
    print(f"Total Time:      {total_time:.2f} seconds")
    print(f"Achieved RPS:    {achieved_rps:.2f} req/sec")
    print(f"Average Latency: {avg_latency_ms:.2f} ms")
    print("Status Code Histogram:")
    for code, count in sorted(status_counts.items()):
        status_label = f"HTTP {code}" if code > 0 else "Connection Error"
        pct = (count / args.requests) * 100
        print(f"  {status_label}: {count} ({pct:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    main()
