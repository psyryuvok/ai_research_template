import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, CliSettingsSource, PydanticBaseSettingsSource, SettingsConfigDict

from src.utils.logging_config import get_file_logger, get_safe_path

logger = get_file_logger(__name__, "normal")


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """
    A reusable Pydantic source that reads from a specific YAML file.
    """

    def __init__(self, settings_cls: Type[BaseSettings], yaml_file: Path | str):
        super().__init__(settings_cls)
        self.yaml_file = Path(yaml_file)

    def get_field_value(self, field: Any, field_name: str) -> Tuple[Any, str, bool]:
        # Boilerplate required by ABC
        return None, field_name, False

    def __call__(self) -> Dict[str, Any]:
        if not self.yaml_file.exists():
            logger.warning(f"Config file not found: {self.yaml_file}. Using defaults.")
            return {}

        with open(self.yaml_file, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


# --- Sub-Models ---


class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    username: str
    password: str
    data: str


class FeatureFlags(BaseModel):
    new_ui: bool = False
    beta_access: bool = False


class RepeatabilityConfig(BaseModel):
    seed: int = 42
    PYTHONHASHSEED: int = 0


class SecurityConfig(BaseModel):
    credentials_path: str = ""


class FilesConfig(BaseModel):
    data_path: str


class DataConfig(BaseModel):
    stride_size: int
    time_size: int
    type_seizure_extract: list[str]


class ModelConfig(BaseModel):
    model_name: str
    early_stop: int


class EdgeDeployment(BaseModel):
    export_tflite: bool = False
    export_onnx: bool = False
    quantization: str = "int8"
    calibration_samples: int = 100


class GithubConfig(BaseModel):
    MY_GOOGLE_DRIVE_PATH: str
    GIT_USERNAME: str
    GIT_REPOSITORY: str


class OutputConfig(BaseModel):
    generate_files: bool
    generate_statistics: bool
    save_images: bool


class OptunaParameters(BaseModel):
    study_name: str
    model_name: list[str]
    batch_size: list[int]
    num_units: list[int]
    dropout_rate: list[float]
    learning_rate: list[float]
    n_trials: int
    timeout: int


class ModelSeizureConfig(BaseModel):
    optuna_parameters: OptunaParameters
    epochs: int


# --- The Main Settings Class ---
class Settings(BaseSettings):
    """
    Application-wide settings managed via Pydantic and YAML/Env sources.
    """

    name: str
    security: SecurityConfig
    files: FilesConfig
    data: DataConfig
    model: ModelConfig
    model_seizure: ModelSeizureConfig
    edge_deployment: EdgeDeployment
    github: GithubConfig
    output: OutputConfig
    repeatability: RepeatabilityConfig = Field(default_factory=RepeatabilityConfig)
    db: DatabaseConfig
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    db_data: str | None = None
    generated: dict | None = None
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        # frozen=True,
        cli_kebab_case=True,
        env_nested_delimiter="___",  # 3 underscores
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:

        is_test_or_jupyter = False
        if sys.argv:
            prog_name = sys.argv[0].lower()
            if "pytest" in prog_name or "ipykernel" in prog_name:
                is_test_or_jupyter = True

        cli_settings: PydanticBaseSettingsSource = CliSettingsSource(
            settings_cls,
            cli_parse_args=not is_test_or_jupyter,
            cli_ignore_unknown_args=True,
        )

        # Manually extract custom YAML path from CLI args
        cli_yaml_path = None
        if not is_test_or_jupyter:
            for i, arg in enumerate(sys.argv):
                if arg in ("--config", "--yaml-file") and i + 1 < len(sys.argv):
                    cli_yaml_path = sys.argv[i + 1]

        init_kwargs = getattr(init_settings, "init_kwargs", {})
        target_file = (
            init_kwargs.get("_yaml_file")
            or cli_yaml_path
            or os.environ.get("ENV_PARAMS")
            or dotenv_values(".env").get("ENV_PARAMS")
            or "config/runs/env_params.yaml"
        )

        target_file = get_safe_path(target_file)

        return (
            init_settings,
            cli_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, target_file),
            file_secret_settings,
        )

    # Post-init hook
    def model_post_init(self, __context: Any) -> None:
        # if you want to sync nested data to flat fields
        if self.db_data is None and self.db:
            self.db_data = self.db.data

    def save_run_config(self, pipeline_name: str | None = None) -> Path | None:
        """
        Explicitly saves the fully resolved configuration to a YAML file for reproducibility.
        If pipeline_name is None, it automatically extracts the name of the calling file.
        """
        if not self.output.generate_files:
            return None

        from datetime import datetime

        # Automatically determine the caller's filename if not provided
        if pipeline_name is None:
            import inspect

            try:
                # stack()[1] is the frame that called this method
                caller_frame = inspect.stack()[1]
                stem = Path(caller_frame.filename).stem
                # Sanitize the filename
                pipeline_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem).strip("_")
            except Exception:
                pipeline_name = "main"

        run_dir = Path("reports/runs") / self.name
        run_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = run_dir / f"run_{pipeline_name}_{timestamp}.yaml"

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
            logger.info(f"Run configuration saved to {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Failed to save run configuration: {e}")
            return None


_settings_instance = None
_settings_lock = threading.Lock()  # Prevents race conditions during startup


def get_settings(force_reload: bool = False, use_singleton: bool = True, **kwargs) -> Settings:
    """
    Returns a Settings instance.

    Args:
        force_reload: If True, re-evaluates all sources (YAML, CLI) and overrides the singleton.
        use_singleton: If True (default), updates and returns the global singleton.
                       If False, returns a completely independent, standalone Settings instance.
        **kwargs: Programmatic overrides.
    """
    global _settings_instance

    # CASE 1: We want a completely independent instance (e.g. for a specific file/test)
    if not use_singleton:
        return Settings(**kwargs)

    # CASE 2: We want to use or update the global singleton
    with _settings_lock:
        if _settings_instance is None or force_reload:
            _settings_instance = Settings(**kwargs)

        elif kwargs:
            # If the singleton exists, and someone tries to pass kwargs WITHOUT
            # force_reload, we raise an error to stop State Clobbering.
            raise ValueError(
                "Settings singleton is already initialized! "
                "You cannot pass kwargs to get_settings() after initialization "
                "unless you also pass force_reload=True or use_singleton=False."
            )

        return _settings_instance
