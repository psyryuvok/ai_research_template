import os
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

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
        # 1. Check Constructor Init (so we can pass _yaml_file in tests)
        # 2. Check System Env Vars
        # 3. Check .env file
        # 4. Fallback to default

        init_kwargs = getattr(init_settings, "init_kwargs", {})
        target_file = init_kwargs.get("_yaml_file") or os.environ.get("ENV_PARAMS") or dotenv_values(".env").get("ENV_PARAMS") or "env_params.yaml"
        target_file = get_safe_path(target_file)
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, target_file),
            file_secret_settings,
        )

    # Use a post-init hook if you want to sync nested data to flat fields
    def model_post_init(self, __context: Any) -> None:
        if self.db_data is None and self.db:
            self.db_data = self.db.data
