"""conftest.py — Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from sentiment.config import AppConfig, ModelConfig, TrainingConfig
from sentiment.data.preprocessing import SentimentTokenizer
from sentiment.models.gru import GRUSentimentModel
from sentiment.serving.predictor import Predictor

SAMPLE_REVIEWS = [
    "This film was absolutely brilliant and moving.",
    "Terrible waste of time. Walked out after 20 minutes.",
    "A perfectly average thriller with highs and lows.",
] * 20


@pytest.fixture(scope="session")
def default_cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.data.vocab_size       = 500
    cfg.data.max_sequence_len = 50
    cfg.model.gru_units       = [16]
    cfg.model.dense_units     = [8]
    cfg.training.epochs       = 1
    cfg.training.batch_size   = 4
    return cfg


@pytest.fixture(scope="session")
def fitted_tokenizer(default_cfg: AppConfig) -> SentimentTokenizer:
    tok = SentimentTokenizer(vocab_size=default_cfg.data.vocab_size, max_len=default_cfg.data.max_sequence_len)
    tok.fit(SAMPLE_REVIEWS)
    return tok


@pytest.fixture(scope="session")
def built_model(default_cfg: AppConfig) -> tf.keras.Model:
    return GRUSentimentModel(default_cfg.model).build(
        vocab_size=default_cfg.data.vocab_size,
        max_len=default_cfg.data.max_sequence_len,
        training_cfg=default_cfg.training,
    )


@pytest.fixture(scope="session")
def predictor(built_model: tf.keras.Model, fitted_tokenizer: SentimentTokenizer) -> Predictor:
    return Predictor(model=built_model, tokenizer=fitted_tokenizer)
