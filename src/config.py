"""
Pydantic Configuration Loader
Centralizes all configuration from config.yaml with type safety and validation.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Literal
from functools import lru_cache

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseModel):
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    models: str = "models"
    reports: str = "reports"
    logs: str = "logs"
    config: str = "config"

    def resolve(self, root: Path) -> "ResolvedPaths":
        return ResolvedPaths(
            data_raw=root / self.data_raw,
            data_processed=root / self.data_processed,
            models=root / self.models,
            reports=root / self.reports,
            logs=root / self.logs,
            config=root / self.config,
        )


class ResolvedPaths:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, Path(v).resolve())


class CICIoT2023Config(BaseModel):
    raw_dir: str = "ciciot2023/MERGED_CSV"
    processed_dir: str = "ciciot2023"
    selected_features_file: str = "selected_features.txt"
    imputation_stats_file: str = "numeric_imputation_stats.json"
    split_metadata_file: str = "split_metadata.json"
    forecast_temporal_meta_file: str = "forecast_temporal_meta.json"
    n_features: int = 36
    test_size: float = 0.2
    random_state: int = 42
    attack_categories: int = 9


class CICIDS2017Config(BaseModel):
    raw_dir: str = "cicids2017"
    processed_dir: str = "cicids2017"
    feature_schema_file: str = "ForecastFeatureSchema.json"
    states_file_template: str = "states_{window_sec}s.csv"
    scaler_file: str = "scaler.joblib"
    window_seconds: int = 60
    sequence_length: int = 10
    forecast_horizon: int = 5
    state_dim: int = 21
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15

    @field_validator("train_split", "val_split", "test_split")
    @classmethod
    def splits_sum_to_one(cls, v, info):
        # Can't validate sum here without all three; validated in model_post_init
        return v

    def model_post_init(self, __context):
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Splits must sum to 1.0, got {total}")


class ModelsConfig(BaseModel):
    xgboost: str = "xgboost_model.joblib"
    random_forest: str = "random_forest_model.joblib"
    logistic_regression: str = "logistic_regression_model.joblib"
    gru_60s: str = "cicids_gru_W10_K5_60s.pt"
    gru_30s: str = "cicids_gru_W10_K5_30s.pt"
    lstm_60s: str = "cicids_lstm_W10_K5_60s.pt"
    lstm_30s: str = "cicids_lstm_W10_K5_30s.pt"


class ModelIntegrityConfig(BaseModel):
    """Expected SHA-256 hashes of trusted model artifacts."""
    xgboost: str = "539a5d661eb36863673efdebe0666d74416031850160c56eddb442a3cc4e0984"
    random_forest: str = "08ed6ed2104e3c54af181b49fcd622dc66ea967811e4bac74fd8425786f1da2c"
    logistic_regression: str = "8fe753b158853f8381db14f70995a4dc005bcb5f66977e8030c844597c30c9f1"
    gru_60s: str = "f56c9e09a42002fdce71a237fbe8b4d053ef46b45337b98c2cad0b39dba40f34"


class XGBoostConfig(BaseModel):
    n_estimators: int = 200
    max_depth: int = 8
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    n_jobs: int = -1


class RandomForestConfig(BaseModel):
    n_estimators: int = 200
    max_depth: int = 15
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    random_state: int = 42
    n_jobs: int = -1


class LogisticRegressionConfig(BaseModel):
    C: float = 1.0
    max_iter: int = 1000
    solver: str = "lbfgs"
    random_state: int = 42


class GRUConfig(BaseModel):
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 50
    patience: int = 10


class LSTMConfig(BaseModel):
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 50
    patience: int = 10


class ForecastingConfig(BaseModel):
    gru: GRUConfig = GRUConfig()
    lstm: LSTMConfig = LSTMConfig()


class TrainingConfig(BaseModel):
    xgboost: XGBoostConfig = XGBoostConfig()
    random_forest: RandomForestConfig = RandomForestConfig()
    logistic_regression: LogisticRegressionConfig = LogisticRegressionConfig()
    forecasting: ForecastingConfig = ForecastingConfig()


class InferenceConfig(BaseModel):
    confidence_threshold: float = 0.5
    batch_size: int = 1024
    device: Literal["auto", "cpu", "cuda"] = "auto"


class DashboardConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8501
    theme: str = "dark"
    max_upload_size_mb: int = 200


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: Literal["json", "text"] = "json"
    file: str = "logs/app.log"
    max_bytes: int = 10485760
    backup_count: int = 5


class CIConfig(BaseModel):
    python_version: str = "3.11"
    test_timeout_seconds: int = 300
    mypy_strict_modules: list[str] = ["src.forecasting", "src.inference"]


class ProjectConfig(BaseModel):
    name: str = "AI-Network-Intrusion-Detection-V2"
    version: str = "0.1.0"
    paths: PathsConfig = PathsConfig()
    ciciot2023: CICIoT2023Config = CICIoT2023Config()
    cicids2017: CICIDS2017Config = CICIDS2017Config()
    models: ModelsConfig = ModelsConfig()
    model_integrity: ModelIntegrityConfig = ModelIntegrityConfig()
    training: TrainingConfig = TrainingConfig()
    inference: InferenceConfig = InferenceConfig()
    dashboard: DashboardConfig = DashboardConfig()
    logging: LoggingConfig = LoggingConfig()
    ci: CIConfig = CIConfig()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    # Environment overrides (all optional)
    paths__data_raw: str | None = None
    paths__data_processed: str | None = None
    paths__models: str | None = None
    paths__reports: str | None = None
    paths__logs: str | None = None
    paths__config: str | None = None

    cicids2017__window_seconds: int | None = None
    cicids2017__sequence_length: int | None = None
    cicids2017__forecast_horizon: int | None = None

    training__forecasting__gru__learning_rate: float | None = None
    training__forecasting__gru__epochs: int | None = None
    training__forecasting__gru__batch_size: int | None = None

    inference__device: Literal["auto", "cpu", "cuda"] | None = None
    inference__confidence_threshold: float | None = None

    dashboard__port: int | None = None
    dashboard__host: str | None = None

    logging__level: str | None = None


def load_config(config_path: str | Path | None = None) -> ProjectConfig:
    """
    Load configuration from config.yaml with environment variable overrides.
    
    Args:
        config_path: Path to config.yaml. If None, searches standard locations.
    
    Returns:
        Validated ProjectConfig instance.
    """
    # Find config file
    if config_path is None:
        for candidate in [
            Path.cwd() / "config.yaml",
            Path(__file__).parent.parent / "config.yaml",
            Path(os.getenv("CONFIG_PATH", "")),
        ]:
            if candidate and candidate.exists():
                config_path = candidate
                break
        else:
            raise FileNotFoundError("config.yaml not found in standard locations")

    config_path = Path(config_path)
    with config_path.open("r") as f:
        raw = yaml.safe_load(f)

    # Apply environment overrides via Settings
    settings = Settings()
    env_overrides = settings.model_dump(exclude_none=True)

    # Deep merge env overrides into raw config
    def deep_merge(base: dict, override: dict) -> dict:
        result = base.copy()
        for k, v in override.items():
            if "__" in k:
                parts = k.split("__")
                target = result
                for p in parts[:-1]:
                    target = target.setdefault(p, {})
                target[parts[-1]] = v
            else:
                result[k] = v
        return result

    merged = deep_merge(raw, env_overrides)
    return ProjectConfig(**merged)


@lru_cache(maxsize=1)
def get_config(config_path: str | Path | None = None) -> ProjectConfig:
    """Cached singleton config instance."""
    return load_config(config_path)


def get_resolved_paths(config: ProjectConfig | None = None) -> ResolvedPaths:
    """Get resolved absolute paths from config."""
    if config is None:
        config = get_config()
    return config.paths.resolve(Path.cwd())


# Convenience functions for common config sections
def get_ciciot2023_config(config: ProjectConfig | None = None) -> CICIoT2023Config:
    if config is None:
        config = get_config()
    return config.ciciot2023


def get_cicids2017_config(config: ProjectConfig | None = None) -> CICIDS2017Config:
    if config is None:
        config = get_config()
    return config.cicids2017


def get_training_config(config: ProjectConfig | None = None) -> TrainingConfig:
    if config is None:
        config = get_config()
    return config.training


def get_inference_config(config: ProjectConfig | None = None) -> InferenceConfig:
    if config is None:
        config = get_config()
    return config.inference


def get_dashboard_config(config: ProjectConfig | None = None) -> DashboardConfig:
    if config is None:
        config = get_config()
    return config.dashboard


def get_model_integrity_config(config: ProjectConfig | None = None) -> ModelIntegrityConfig:
    if config is None:
        config = get_config()
    return config.model_integrity


if __name__ == "__main__":
    # Quick validation
    cfg = get_config()
    print(f"Project: {cfg.name} v{cfg.version}")
    print(f"Paths: {cfg.paths}")
    print(f"CICIDS-2017 window: {cfg.cicids2017.window_seconds}s")
    print(f"GRU hidden: {cfg.training.forecasting.gru.hidden_size}")
    print("✅ Config loaded successfully")