import os
import yaml
import pytest
from pathlib import Path
from pydantic import ValidationError

# Import the refactored Settings class from your main file
# Assuming your file is named 'config.py'
from src.utils.settings import Settings, DatabaseConfig

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
    if "APP_NAME" in os.environ:
        del os.environ["APP_NAME"]
    if "DB__PASSWORD" in os.environ:
        del os.environ["DB__PASSWORD"]

    yield

    # Restore environment
    os.environ.clear()
    os.environ.update(old_environ)


@pytest.fixture
def yaml_config_file(tmp_path):
    """
    Creates a temporary YAML file for testing and returns its path.
    Using 'tmp_path' ensures cleanup happens automatically.
    """
    config_data = {
        "app_name": "YAML App",
        "db": {"host": "yaml_host", "port": 9090, "username": "yaml_user", "password": "yaml_password", "data": "yaml_data"},
        "features": {"new_ui": True},
    }

    config_file = tmp_path / "test_config.yaml"
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

    assert settings.app_name == "YAML App"
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
    os.environ["APP_NAME"] = "Env App"
    os.environ["DB___PASSWORD"] = "env_password"  # Double underscore for nesting

    settings = Settings(_yaml_file=yaml_config_file)

    # APP_NAME should come from Env, not YAML
    assert settings.app_name == "Env App"
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
    os.environ["DB___USERNAME"] = "env_user"
    os.environ["DB___PASSWORD"] = "env_pass"
    os.environ["DB___DATA"] = "env_data"

    # Point to a non-existent file
    settings = Settings(_yaml_file="non_existent.yaml")

    assert settings.app_name == "Default App"  # Fallback to default in class
    assert settings.db.username == "env_user"


def test_validation_error(clean_env, yaml_config_file, tmp_path):
    """
    Test 4: Verify Pydantic raises an error if types are wrong in YAML.
    """
    bad_data = {
        "db": {
            "port": "not_a_number",  # Should be int
            "username": "u",
            "password": "p",
            "data": "d",
        }
    }
    bad_file = tmp_path / "bad_config.yaml"
    with open(bad_file, "w") as f:
        yaml.dump(bad_data, f)

    with pytest.raises(ValidationError) as excinfo:
        Settings(_yaml_file=str(bad_file))

    assert "Input should be a valid integer" in str(excinfo.value)


def test_init_kwargs_precedence(clean_env, yaml_config_file):
    """
    Test 5: Verify that passing arguments directly to __init__ overrides everything.
    """
    os.environ["APP_NAME"] = "Env App"

    # Pass app_name directly to constructor
    settings = Settings(_yaml_file=yaml_config_file, app_name="Init App")

    assert settings.app_name == "Init App"
