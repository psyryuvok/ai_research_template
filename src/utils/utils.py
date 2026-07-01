import asyncio
import os
import random
import shutil
import time
import uuid
from datetime import datetime
from functools import wraps
from typing import Any, Callable

import mlflow
import numpy as np
import tensorflow as tf
import yaml

from src.utils.logging_config import get_file_logger
from src.utils.settings import RepeatabilityConfig

logger = get_file_logger(__name__, "normal")


def disable_randomness(repeatability: RepeatabilityConfig):
    os.environ["PYTHONHASHSEED"] = str(repeatability.PYTHONHASHSEED)
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

    # Use HTTP tracking URI so that artifact proxying (mlflow-artifacts:/) works
    mlflow.set_tracking_uri("http://localhost:8089")
    # When serving artifacts via the MLflow server in Docker, the server needs to proxy the requests
    artifact_location = "mlflow-artifacts:/"

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name, artifact_location=artifact_location)


with open("config/runs/env_params.yaml", "r") as get_config_yaml:  # "../../config/runs/env_params.yaml"
    config_variables = yaml.safe_load(get_config_yaml)


class UniversalTimer:
    """
    A versatile timer that can be used as a context manager (sync/async)
    or as a decorator for sync and async functions.
    """

    def __init__(self, label: str):
        """
        Initializes the timer with a label.

        Args:
            label (str): The name to associate with this timing block.
        """
        self.label = label
        self.start_time: float = 0.0
        self.run_id = uuid.uuid4().hex[:6]

    def _format_args(self, *args: Any, **kwargs: Any) -> str:
        """
        Heuristic to join args and kwargs into a readable string.

        Args:
            *args: Positional arguments to format.
            **kwargs: Keyword arguments to format.

        Returns:
            str: A formatted string representation of the arguments.
        """
        arg_str = ", ".join(map(repr, args))
        kwarg_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        combined = ", ".join(filter(None, [arg_str, kwarg_str]))
        return f"({combined})" if combined else "()"

    def _log_start(self, params: str = "") -> None:
        """
        Logs the start of the timed block.

        Args:
            params (str): Additional parameters to include in the log.
        """
        self.start_time = time.perf_counter()
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        logger.info(f"START [{self.label}] [{self.run_id}] at {timestamp} | Args: {params}")

    def _log_duration(self) -> None:
        """Logs the total duration since start_time."""
        duration = time.perf_counter() - self.start_time
        logger.info(f"END   [{self.label}] [{self.run_id}] : {duration:.4f}s")

    # --- Context Manager (Manual use won't have func params) ---
    def __enter__(self) -> "UniversalTimer":
        """Starts timing when entering a context."""
        self._log_start("manual-context")
        return self

    def __exit__(self, *args: Any) -> None:
        """Logs duration when exiting a context."""
        self._log_duration()

    async def __aenter__(self) -> "UniversalTimer":
        """Starts timing when entering an async context."""
        self._log_start("manual-async-context")
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Logs duration when exiting an async context."""
        self._log_duration()

    # --- Decorator Support ---
    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorates a function to log its execution time.

        Args:
            func (Callable): The function to decorate.

        Returns:
            Callable: The wrapped function.
        """
        label = self.label
        params_func = self._format_args  # Reference to helper

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                instance = UniversalTimer(label)
                instance._log_start(params_func(*args, **kwargs))
                try:
                    return await func(*args, **kwargs)
                finally:
                    instance._log_duration()

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                instance = UniversalTimer(label)
                instance._log_start(params_func(*args, **kwargs))
                try:
                    return func(*args, **kwargs)
                finally:
                    instance._log_duration()

            return sync_wrapper
