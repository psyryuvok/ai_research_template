import os
import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

# Import the refactored Settings class from your main file
# Assuming your file is named 'config.py'
from src.utils.settings import DatabaseConfig, Settings

# --- Fixtures ---


@pytest.fixture
def clean_env():
    """
    Ensures environment variables are cleared before and after each test
    to prevent side effects.
    """
    old_environ = dict(os.environ)
    # Clear relevant keys
    if "ENV_PARAMS" in os.environ:
        del os.environ["ENV_PARAMS"]
    if "NAME" in os.environ:
        del os.environ["NAME"]
    if "DB__PASSWORD" in os.environ:
        del os.environ["DB__PASSWORD"]

    yield

    # Restore environment
    os.environ.clear()
    os.environ.update(old_environ)


@pytest.fixture
def workspace_tmp_path():
    """
    Creates a temporary directory inside the tests folder for configuration files
    to avoid path traversal blocks from get_safe_path.
    """
    tmp_dir = Path(__file__).parent / "tmp_workspace"
    tmp_dir.mkdir(exist_ok=True)
    yield tmp_dir
    # shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def yaml_config_file(workspace_tmp_path):
    """
    Creates a temporary YAML file for testing and returns its path.
    Using 'workspace_tmp_path' ensures it stays inside the project root.
    """
    config_data = {
        "name": "YAML App",
        "security": {"credentials_path": ""},
        "files": {"data_path": ""},
        "data": {"stride_size": 9, "time_size": 6, "type_seizure_extract": ["false_alarms"]},
        "model": {"model_name": "test", "early_stop": 5},
        "model_seizure": {
            "optuna_parameters": {
                "study_name": "s",
                "model_name": ["m1"],
                "learning_rate": [0.1, 0.2],
                "num_units": [1, 2],
                "dropout_rate": [0.1, 0.2],
                "batch_size": [1, 2],
                "n_trials": 1,
                "timeout": 1,
            },
            "epochs": 1,
        },
        "github": {"MY_GOOGLE_DRIVE_PATH": "", "GIT_USERNAME": "", "GIT_REPOSITORY": ""},
        "output": {"generate_files": False, "generate_statistics": False, "save_images": False},
        "db": {"host": "yaml_host", "port": 9090, "username": "yaml_user", "password": "yaml_password", "data": "yaml_data"},
        "features": {"new_ui": True},
    }

    config_file = workspace_tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    return str(config_file)


# --- Tests ---


def test_load_from_yaml(clean_env, yaml_config_file):
    """
    Test 1: Verify values are loaded correctly from the YAML file.
    """
    # We inject the specific yaml file path using our custom logic
    settings = Settings(_yaml_file=yaml_config_file)

    assert settings.name == "YAML App"
    assert settings.db.host == "yaml_host"
    assert settings.db.port == 9090
    assert settings.features.new_ui is True
    # Verify the post_init hook worked
    assert settings.db_data == "yaml_data"


def test_env_var_precedence(clean_env, yaml_config_file):
    """
    Test 2: Verify that System ENV vars override YAML values.
    Pydantic default priority: Init > Env > DotEnv > YAML > Defaults
    """
    os.environ["NAME"] = "Env App"
    os.environ["DB___PASSWORD"] = "env_password"  # Double underscore for nesting

    settings = Settings(_yaml_file=yaml_config_file)

    # NAME should come from Env, not YAML
    assert settings.name == "Env App"
    # DB Password should come from Env
    assert settings.db.password == "env_password"
    # DB Host should still come from YAML (since Env didn't set it)
    assert settings.db.host == "yaml_host"


def test_missing_yaml_file(clean_env):
    """
    Test 3: If YAML file is missing, it should not crash, but use defaults/env.
    However, required fields (like db info) must still be present or it raises ValidationError.
    """
    # We must provide required fields via ENV since YAML is missing
    os.environ["NAME"] = "Env App"
    os.environ["SECURITY___CREDENTIALS_PATH"] = "c"
    os.environ["FILES___DATA_PATH"] = "d"
    os.environ["DATA___STRIDE_SIZE"] = "9"
    os.environ["DATA___TIME_SIZE"] = "6"
    os.environ["DATA___TYPE_SEIZURE_EXTRACT"] = '["false_alarms"]'
    os.environ["MODEL___MODEL_NAME"] = "test"
    os.environ["MODEL___EARLY_STOP"] = "5"
    os.environ["MODEL_SEIZURE___OPTUNA_PARAMETERS___STUDY_NAME"] = "s"
    os.environ["MODEL_SEIZURE___OPTUNA_PARAMETERS___MODEL_NAME"] = '["m1"]'
    os.environ["MODEL_SEIZURE___OPTUNA_PARAMETERS___LEARNING_RATE"] = "[0.1, 0.2]"
    os.environ["MODEL_SEIZURE___OPTUNA_PARAMETERS___NUM_UNITS"] = "[1, 2]"
    os.environ["MODEL_SEIZURE___OPTUNA_PARAMETERS___DROPOUT_RATE"] = "[0.1, 0.2]"
    os.environ["MODEL_SEIZURE___OPTUNA_PARAMETERS___BATCH_SIZE"] = "[1, 2]"
    os.environ["MODEL_SEIZURE___OPTUNA_PARAMETERS___N_TRIALS"] = "1"
    os.environ["MODEL_SEIZURE___OPTUNA_PARAMETERS___TIMEOUT"] = "1"
    os.environ["MODEL_SEIZURE___EPOCHS"] = "1"
    os.environ["GITHUB___MY_GOOGLE_DRIVE_PATH"] = ""
    os.environ["GITHUB___GIT_USERNAME"] = ""
    os.environ["GITHUB___GIT_REPOSITORY"] = ""
    os.environ["OUTPUT___GENERATE_FILES"] = "False"
    os.environ["OUTPUT___GENERATE_STATISTICS"] = "False"
    os.environ["OUTPUT___SAVE_IMAGES"] = "False"

    os.environ["DB___USERNAME"] = "env_user"
    os.environ["DB___PASSWORD"] = "env_pass"
    os.environ["DB___DATA"] = "env_data"

    # Point to a non-existent file
    settings = Settings(_yaml_file="non_existent.yaml")

    assert settings.name == "Env App"  # Fallback to env
    assert settings.db.username == "env_user"


def test_validation_error(clean_env, yaml_config_file, workspace_tmp_path):
    """
    Test 4: Verify Pydantic raises an error if types are wrong in YAML.
    """
    bad_data = {
        "name": "YAML App",
        "security": {"credentials_path": ""},
        "files": {"data_path": ""},
        "data": {"stride_size": 9, "time_size": 6, "type_seizure_extract": ["false_alarms"]},
        "model": {"model_name": "test", "early_stop": 5},
        "model_seizure": {
            "optuna_parameters": {
                "study_name": "s",
                "model_name": ["m1"],
                "learning_rate": [0.1, 0.2],
                "num_units": [1, 2],
                "dropout_rate": [0.1, 0.2],
                "batch_size": [1, 2],
                "n_trials": 1,
                "timeout": 1,
            },
            "epochs": 1,
        },
        "github": {"MY_GOOGLE_DRIVE_PATH": "", "GIT_USERNAME": "", "GIT_REPOSITORY": ""},
        "output": {"generate_files": False, "generate_statistics": False, "save_images": False},
        "db": {
            "port": "not_a_number",  # Should be int
            "username": "u",
            "password": "p",
            "data": "d",
        },
    }
    bad_file = workspace_tmp_path / "bad_config.yaml"
    with open(bad_file, "w") as f:
        yaml.dump(bad_data, f)

    with pytest.raises(ValidationError) as excinfo:
        Settings(_yaml_file=str(bad_file))

    assert "Input should be a valid integer" in str(excinfo.value)


def test_init_kwargs_precedence(clean_env, yaml_config_file):
    """
    Test 5: Verify that passing arguments directly to __init__ overrides everything.
    """
    os.environ["NAME"] = "Env App"

    # Pass name directly to constructor
    settings = Settings(_yaml_file=yaml_config_file, name="Init App")

    assert settings.name == "Init App"
