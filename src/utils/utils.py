import os
import random
import shutil
import time
import asyncio
from functools import wraps

import mlflow
import numpy as np
import tensorflow as tf
import yaml
from src.utils.logging_config import get_file_logger
from src.utils.settings import RepeatabilityConfig

logger = get_file_logger(__name__, "normal")


def disable_randomness(repeatability: RepeatabilityConfig):
    os.environ['PYTHONHASHSEED'] = str(repeatability.PYTHONHASHSEED)
    np.random.seed(repeatability.PYTHONHASHSEED)
    random.seed(repeatability.PYTHONHASHSEED)
    tf.random.set_seed(repeatability.PYTHONHASHSEED)


def get_or_create_mflow_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    db_path = os.path.abspath("reports")
    db_file = os.path.join(db_path, "mlflow.db")
    backup_file = f"{db_file}.backup"

    # Restore from backup if it exists to ensure local training uses absolute paths
    if os.path.exists(backup_file):
        logger.info("Restoring MLflow DB from backup for local training.")
        shutil.copy2(backup_file, db_file)
        os.remove(backup_file)

    mlflow.set_tracking_uri(f"sqlite:///{db_file}")
    artifact_location = f"file://{db_path}/mlartifacts"

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name, artifact_location=artifact_location)


with open("config/runs/env_params.yaml", "r") as get_config_yaml:  # "../../config/runs/env_params.yaml"
    config_variables = yaml.safe_load(get_config_yaml)


class UniversalTimer:
    def __init__(self, label: str):
        self.label = label
        self.start: float

    # --- Context Manager Support ---
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self._log_duration()

    # --- Async Context Manager Support ---
    async def __aenter__(self):
        self.start = time.perf_counter()
        return self

    async def __aexit__(self, *args):
        self._log_duration()

    def _log_duration(self):
        duration = time.perf_counter() - self.start
        logger.info(f"Timer [{self.label}]: {duration:.4f}s")

    # --- Decorator Support ---
    def __call__(self, func):
        label = self.label

        # 1. Handle Async Functions
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Create a NEW instance to ensure concurrency safety
                async with UniversalTimer(label):
                    return await func(*args, **kwargs)

            return async_wrapper

        # 2. Handle Sync Functions
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Create a NEW instance to ensure concurrency safety
                with UniversalTimer(label):
                    return func(*args, **kwargs)

            return sync_wrapper
