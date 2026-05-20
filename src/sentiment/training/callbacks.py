"""
callbacks.py — Custom Keras callbacks for production training.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import mlflow
import tensorflow as tf

log = logging.getLogger(__name__)


class MLflowCallback(tf.keras.callbacks.Callback):
    """Log epoch metrics + model artifact to MLflow."""

    def __init__(self, log_model: bool = True) -> None:
        super().__init__()
        self.log_model = log_model

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if logs:
            mlflow.log_metrics(logs, step=epoch)

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        if self.log_model and self.model is not None:
            mlflow.keras.log_model(self.model, artifact_path="model")
            log.info("Model logged to MLflow.")


class WarmupScheduler(tf.keras.callbacks.Callback):
    """Linear LR warm-up for the first `warmup_steps` batches."""

    def __init__(self, warmup_steps: int, target_lr: float) -> None:
        super().__init__()
        self.warmup_steps = warmup_steps
        self.target_lr    = target_lr
        self._step        = 0

    def on_train_batch_begin(self, batch: int, logs: Any = None) -> None:
        if self._step < self.warmup_steps:
            lr = self.target_lr * (self._step + 1) / self.warmup_steps
            learning_rate = self.model.optimizer.learning_rate
            if hasattr(learning_rate, "assign"):
                learning_rate.assign(lr)
            else:
                self.model.optimizer.learning_rate = lr
        self._step += 1


class EpochTimingCallback(tf.keras.callbacks.Callback):
    """Log wall-clock time per epoch."""

    def on_epoch_begin(self, epoch: int, logs: Any = None) -> None:
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
        elapsed = time.perf_counter() - self._t0
        log.info("Epoch %d — wall time: %.1fs", epoch + 1, elapsed)
        try:
            mlflow.log_metric("epoch_time_sec", elapsed, step=epoch)
        except Exception:
            pass
