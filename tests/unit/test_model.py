"""Unit tests for GRU model construction and forward pass."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from sentiment.config import ModelConfig, TrainingConfig
from sentiment.exceptions import ModelError
from sentiment.models.factory import ModelRegistry
from sentiment.models.gru import BahdanauAttention, GRUSentimentModel


class TestBahdanauAttention:
    def test_output_shape(self) -> None:
        H   = tf.random.normal((8, 20, 64))
        out = BahdanauAttention(units=32)(H)
        assert out.shape == (8, 64)


class TestGRUSentimentModel:
    @pytest.fixture
    def small_model(self) -> tf.keras.Model:
        cfg             = ModelConfig()
        cfg.gru_units   = [16]
        cfg.dense_units = [8]
        cfg.attention   = True
        return GRUSentimentModel(cfg).build(vocab_size=500, max_len=50)

    def test_output_shape(self, small_model: tf.keras.Model) -> None:
        X = np.random.randint(0, 500, (4, 50))
        assert small_model.predict(X, verbose=0).shape == (4, 1)

    def test_output_in_01(self, small_model: tf.keras.Model) -> None:
        X   = np.random.randint(0, 500, (4, 50))
        out = small_model.predict(X, verbose=0).flatten()
        assert np.all((out >= 0) & (out <= 1))

    def test_registry_resolves(self) -> None:
        assert isinstance(ModelRegistry.create(ModelConfig()), GRUSentimentModel)

    def test_unknown_model_raises(self) -> None:
        cfg      = ModelConfig()
        cfg.name = "does_not_exist"
        with pytest.raises(ModelError):
            ModelRegistry.create(cfg)
