"""
elastic_logger.py
-----------------
Sends MLflow run metrics, model parameters, system metrics,
and Docker container stats to Elasticsearch for Kibana monitoring.

Index mappings:
  - mlflow-metrics : one doc per MLflow training run
  - system-metrics : one doc per system snapshot (CPU, RAM, disk)
  - docker-metrics  : one doc per running container stats snapshot
"""

import datetime
import os

import psutil

try:
    import docker as docker_sdk

    DOCKER_AVAILABLE = True
except Exception:
    DOCKER_AVAILABLE = False

from elasticsearch import Elasticsearch, exceptions as es_exc


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------


def get_es_client():
    """Return an Elasticsearch client using ELASTICSEARCH_URL env var."""
    url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    try:
        client = Elasticsearch(url, request_timeout=5)
        if client.ping():
            print(f"✅ Connected to Elasticsearch at {url}")
            return client
        else:
            print(f"⚠️  Elasticsearch ping failed at {url}")
            return None
    except es_exc.ConnectionError as exc:
        print(f"⚠️  Could not connect to Elasticsearch: {exc}")
        return None


# ---------------------------------------------------------------------------
# MLflow run logger
# ---------------------------------------------------------------------------


def log_mlflow_run(run_info: dict):
    """
    Send a single MLflow run document to the 'mlflow-metrics' index.

    Parameters
    ----------
    run_info : dict
        Expected keys:
          run_id, run_name, model_name, params (dict), metrics (dict),
          status, start_time (ISO str)
    """
    client = get_es_client()
    if client is None:
        return

    doc = {
        "@timestamp": run_info.get("start_time", datetime.datetime.utcnow().isoformat()),
        "run_id": run_info.get("run_id", ""),
        "run_name": run_info.get("run_name", ""),
        "model_name": run_info.get("model_name", ""),
        "status": run_info.get("status", "FINISHED"),
        "params": run_info.get("params", {}),
        "metrics": run_info.get("metrics", {}),
        "source": "mlflow",
    }

    try:
        resp = client.index(index="mlflow-metrics", document=doc)
        print(f"📤 MLflow run logged to Elasticsearch: {resp['result']}")
    except Exception as exc:
        print(f"⚠️  Failed to log MLflow run to Elasticsearch: {exc}")


# ---------------------------------------------------------------------------
# System metrics logger
# ---------------------------------------------------------------------------


def log_system_metrics():
    """
    Collect CPU, RAM, and disk usage and push to 'system-metrics' index.
    """
    client = get_es_client()
    if client is None:
        return

    disk = psutil.disk_usage("/")
    doc = {
        "@timestamp": datetime.datetime.utcnow().isoformat(),
        "source": "system",
        "cpu_percent": psutil.cpu_percent(interval=1),
        "ram_total_mb": psutil.virtual_memory().total / (1024 * 1024),
        "ram_used_mb": psutil.virtual_memory().used / (1024 * 1024),
        "ram_percent": psutil.virtual_memory().percent,
        "disk_total_gb": disk.total / (1024 ** 3),
        "disk_used_gb": disk.used / (1024 ** 3),
        "disk_percent": disk.percent,
    }

    try:
        resp = client.index(index="system-metrics", document=doc)
        print(f"📤 System metrics logged to Elasticsearch: {resp['result']}")
    except Exception as exc:
        print(f"⚠️  Failed to log system metrics to Elasticsearch: {exc}")


# ---------------------------------------------------------------------------
# Docker container metrics logger
# ---------------------------------------------------------------------------


def log_docker_metrics():
    """
    Collect running Docker container stats and push to 'docker-metrics' index.
    Silently skipped if the Docker SDK is unavailable.
    """
    if not DOCKER_AVAILABLE:
        print("⚠️  Docker SDK not available — skipping docker metrics logging.")
        return

    client = get_es_client()
    if client is None:
        return

    try:
        dc = docker_sdk.from_env()
        containers = dc.containers.list()
        for container in containers:
            stats = container.stats(stream=False)
            # CPU delta calculation
            cpu_delta = (
                stats["cpu_stats"]["cpu_usage"]["total_usage"]
                - stats["precpu_stats"]["cpu_usage"]["total_usage"]
            )
            sys_delta = (
                stats["cpu_stats"]["system_cpu_usage"]
                - stats["precpu_stats"]["system_cpu_usage"]
            )
            num_cpus = stats["cpu_stats"].get("online_cpus", 1)
            cpu_pct = (cpu_delta / sys_delta) * num_cpus * 100.0 if sys_delta > 0 else 0.0

            # Memory
            mem_usage = stats["memory_stats"].get("usage", 0) / (1024 * 1024)
            mem_limit = stats["memory_stats"].get("limit", 1) / (1024 * 1024)
            mem_pct = (mem_usage / mem_limit) * 100 if mem_limit > 0 else 0.0

            doc = {
                "@timestamp": datetime.datetime.utcnow().isoformat(),
                "source": "docker",
                "container_id": container.short_id,
                "container_name": container.name,
                "image": container.image.tags[0] if container.image.tags else "unknown",
                "status": container.status,
                "cpu_percent": round(cpu_pct, 2),
                "mem_usage_mb": round(mem_usage, 2),
                "mem_limit_mb": round(mem_limit, 2),
                "mem_percent": round(mem_pct, 2),
            }

            resp = client.index(index="docker-metrics", document=doc)
            print(
                f"📤 Docker stats for '{container.name}' logged: {resp['result']}"
            )
    except Exception as exc:
        print(f"⚠️  Failed to log Docker metrics to Elasticsearch: {exc}")


# ---------------------------------------------------------------------------
# CLI entry point — run directly: python elastic_logger.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Logging system metrics ===")
    log_system_metrics()
    print("\n=== Logging Docker container metrics ===")
    log_docker_metrics()
