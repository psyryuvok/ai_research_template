import os
import random

import mlflow
import numpy as np
import tensorflow as tf
import yaml


def disable_randomness(repeatability):
    os.environ['PYTHONHASHSEED'] = repeatability["PYTHONHASHSEED"]
    np.random.seed(repeatability["seed"])
    random.seed(repeatability["seed"])
    tf.random.set_seed(repeatability["seed"])


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

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)


with open("../../config/runs/env_params.yaml", "r") as get_config_yaml:
    config_variables = yaml.safe_load(get_config_yaml)
