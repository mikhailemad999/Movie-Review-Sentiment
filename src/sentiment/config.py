"""
config.py — Typed, validated configuration with Pydantic v2.

All settings flow from YAML files or environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseModel):
    vocab_size:       Annotated[int,   Field(gt=0)]           = 10_000
    max_sequence_len: Annotated[int,   Field(gt=0)]           = 200
    test_split:       Annotated[float, Field(gt=0, lt=1)]     = 0.2
    val_split:        Annotated[float, Field(gt=0, lt=1)]     = 0.1
    random_seed:      int                                      = 42


class ModelConfig(BaseModel):
    name:            str       = "gru_sentiment_v1"
    embedding_dim:   int       = 128
    gru_units:       list[int] = [128, 64]
    attention:       bool      = True
    dense_units:     list[int] = [128, 64]
    dropout_rate:    float     = 0.3
    spatial_dropout: float     = 0.2
    l2_reg:          float     = 1e-4

    @model_validator(mode="after")
    def validate_units(self) -> "ModelConfig":
        if not self.gru_units:
            raise ValueError("gru_units must not be empty")
        return self


class TrainingConfig(BaseModel):
    batch_size:               int   = 64
    epochs:                   int   = 15
    learning_rate:            float = 1e-3
    lr_warmup_steps:          int   = 500
    weight_decay:             float = 0.01
    label_smoothing:          float = 0.1
    early_stopping_patience:  int   = 4
    reduce_lr_patience:       int   = 2
    reduce_lr_factor:         float = 0.5
    min_lr:                   float = 1e-7
    mixed_precision:          bool  = False


class LoggingConfig(BaseModel):
    level:      str  = "INFO"
    mlflow_uri: str  = "http://localhost:5000"
    experiment: str  = "gru-sentiment"
    log_model:  bool = True


class PathsConfig(BaseModel):
    artifacts:   Path = Path("artifacts/")
    model_dir:   Path = Path("artifacts/models/")
    logs_dir:    Path = Path("artifacts/logs/")
    reports_dir: Path = Path("artifacts/reports/")

    @model_validator(mode="after")
    def create_dirs(self) -> "PathsConfig":
        for p in (self.artifacts, self.model_dir, self.logs_dir, self.reports_dir):
            p.mkdir(parents=True, exist_ok=True)
        return self


class AppConfig(BaseModel):
    """Root configuration object — load via AppConfig.from_yaml()."""

    data:     DataConfig     = DataConfig()
    model:    ModelConfig    = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    logging:  LoggingConfig  = LoggingConfig()
    paths:    PathsConfig    = PathsConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    @classmethod
    def from_env(cls) -> "AppConfig":
        cfg_path = os.environ.get("SENTIMENT_CONFIG", "configs/default.yaml")
        return cls.from_yaml(cfg_path)


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SENTIMENT_", env_file=".env")

    host:            str  = "0.0.0.0"
    port:            int  = 8000
    workers:         int  = 1
    log_level:       str  = "info"
    model_path:      Path = Path("artifacts/models/best_model.keras")
    tokenizer_path:  Path = Path("artifacts/models/tokenizer.pkl")
    max_batch:       int  = 64
    request_timeout: float = 30.0
    enable_metrics:  bool = True
