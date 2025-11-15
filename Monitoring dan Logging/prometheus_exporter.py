from prometheus_client import start_http_server, Summary, Counter, Gauge
import requests
import time
import random
import psutil

LATENCY = Summary('request_latency_seconds', 'Time spent processing request')
THROUGHPUT = Counter('request_count_total', 'Total number of requests')
ERROR_COUNT = Counter('request_error_total', 'Total number of failed requests')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes')

MLFLOW_URL = "http://localhost:5000/invocations"  # endpoint MLflow

INPUT_KEYS = [str(i) for i in range(15)]  # "0" hingga "14"

@LATENCY.time()
def call_mlflow():
    THROUGHPUT.inc()
    try:
        inputs = {k: [random.random()] for k in INPUT_KEYS}
        data = {"inputs": inputs}  # format dict of lists sesuai signature MLflow
        r = requests.post(MLFLOW_URL, json=data, timeout=5)
        r.raise_for_status()
        print(f"Request succeeded: {r.json()}")
    except requests.exceptions.RequestException as e:
        ERROR_COUNT.inc()
        print(f"Request failed: {e}")

    MEMORY_USAGE.set(psutil.virtual_memory().used)

if __name__ == "__main__":
    start_http_server(8001)
    print("Prometheus exporter running on port 8001...")
    while True:
        call_mlflow()
        time.sleep(1)
