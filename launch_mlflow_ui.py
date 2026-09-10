"""
Script to launch the local MLflow Tracking Server & UI Dashboard.
Opens the web UI at http://127.0.0.1:5000 to inspect experiments, runs, parameters,
metrics charts, artifacts, and the Model Registry.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import subprocess
import logging
from notebooks.src.config import BASE_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MLflowUILauncher")

MLFLOW_DB_PATH = BASE_DIR / "data" / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = BASE_DIR / "mlruns"


def start_mlflow_ui(port: int = 5000):
    backend_uri = f"sqlite:///{str(MLFLOW_DB_PATH).replace('\\', '/')}"
    artifacts_uri = f"file:///{str(MLFLOW_ARTIFACTS_DIR).replace('\\', '/')}"
    
    logger.info("=" * 70)
    logger.info("       STARTING MLFLOW EXPERIMENT TRACKING & REGISTRY SERVER")
    logger.info("=" * 70)
    logger.info(f"Backend Database Store:  {backend_uri}")
    logger.info(f"Artifacts Directory:     {artifacts_uri}")
    logger.info(f"MLflow Web Dashboard:    http://127.0.0.1:{port}")
    logger.info("Press Ctrl+C to terminate.")
    logger.info("=" * 70)

    cmd = [
        "mlflow", "ui",
        "--backend-store-uri", backend_uri,
        "--default-artifact-root", artifacts_uri,
        "--host", "127.0.0.1",
        "--port", str(port)
    ]
    
    subprocess.run(cmd)


if __name__ == "__main__":
    start_mlflow_ui()
