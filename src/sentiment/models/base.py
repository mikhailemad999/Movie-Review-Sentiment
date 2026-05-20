"""
base.py — Abstract base class for all sentiment models.
"""

from __future__ import annotations

import abc
from pathlib import Path

import tensorflow as tf
from sentiment.config import ModelConfig


class BaseSentimentModel(abc.ABC):

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg    = cfg
        self._model: tf.keras.Model | None = None

    @abc.abstractmethod
    def build(self, vocab_size: int, max_len: int) -> tf.keras.Model: ...

    @property
    def model(self) -> tf.keras.Model:
        if self._model is None:
            raise RuntimeError("Call .build() before accessing .model")
        return self._model

    def summary(self) -> None:
        self.model.summary(line_length=90)

    def save(self, path: str | Path, fmt: str = "tf") -> None:
        self.model.save(str(path), save_format=fmt)

    def predict_proba(self, X: object) -> object:
        return self.model.predict(X, verbose=0)

    def count_params(self) -> int:
        return int(self.model.count_params())
