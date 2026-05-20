"""
trainer.py — Orchestrates the full training lifecycle.
"""

from __future__ import annotations

import logging
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import mlflow
import tensorflow as tf

from sentiment.config import AppConfig
from sentiment.data.dataset import IMDBData, IMDBDataLoader
from sentiment.models.factory import ModelRegistry
from sentiment.training.callbacks import EpochTimingCallback, MLflowCallback, WarmupScheduler

log = logging.getLogger(__name__)


@dataclass
class TrainResult:
    run_id:            str
    model_path:        Path
    tokenizer_path:    Path
    best_val_accuracy: float
    best_val_auc:      float
    history:           dict


class Trainer:
    """
    High-level training orchestrator.

    Usage::

        cfg    = AppConfig.from_yaml("configs/default.yaml")
        result = Trainer(cfg).run()
    """

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._setup_logging()
        self._setup_mixed_precision()

    def run(self) -> TrainResult:
        mlflow.set_tracking_uri(self.cfg.logging.mlflow_uri)
        mlflow.set_experiment(self.cfg.logging.experiment)

        with mlflow.start_run() as run:
            mlflow.log_params(self._flat_params())
            data    = self._load_data()
            model   = self._build_model(data)
            history = self._train(model, data)
            paths   = self._save_artifacts(model, data)

        return TrainResult(
            run_id=run.info.run_id,
            model_path=paths["model"],
            tokenizer_path=paths["tokenizer"],
            best_val_accuracy=max(history.history["val_accuracy"]),
            best_val_auc=max(history.history.get("val_auc", [0.0])),
            history=history.history,
        )

    def _load_data(self) -> IMDBData:
        log.info("Loading dataset ...")
        return IMDBDataLoader(self.cfg.data).load(batch_size=self.cfg.training.batch_size)

    def _build_model(self, data: IMDBData) -> tf.keras.Model:
        log.info("Building model '%s' ...", self.cfg.model.name)
        sentiment_model = ModelRegistry.create(self.cfg.model)
        model = sentiment_model.build(
            vocab_size=data.tokenizer.effective_vocab_size,
            max_len=self.cfg.data.max_sequence_len,
            training_cfg=self.cfg.training,
        )
        model.summary(print_fn=log.info)
        return model

    def _train(self, model: tf.keras.Model, data: IMDBData) -> tf.keras.callbacks.History:
        return model.fit(
            data.train.dataset,
            validation_data=data.val.dataset,
            epochs=self.cfg.training.epochs,
            class_weight=data.class_weights,
            callbacks=self._build_callbacks(),
            verbose=1,
        )

    def _build_callbacks(self) -> list[tf.keras.callbacks.Callback]:
        cfg = self.cfg
        t   = cfg.training
        p   = cfg.paths
        callbacks: list[tf.keras.callbacks.Callback] = [
            WarmupScheduler(t.lr_warmup_steps, t.learning_rate),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=t.early_stopping_patience,
                restore_best_weights=True, mode="max", verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(p.model_dir / "best_model.keras"),
                monitor="val_auc", save_best_only=True, mode="max", verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=t.reduce_lr_factor,
                patience=t.reduce_lr_patience, min_lr=t.min_lr, verbose=1,
            ),
            MLflowCallback(log_model=cfg.logging.log_model),
            EpochTimingCallback(),
        ]
        if importlib.util.find_spec("tensorboard") is not None:
            callbacks.insert(-2, tf.keras.callbacks.TensorBoard(log_dir=str(p.logs_dir), histogram_freq=1))
        else:
            log.warning("TensorBoard is not installed; skipping TensorBoard callback.")
        return callbacks

    def _save_artifacts(self, model: tf.keras.Model, data: IMDBData) -> dict[str, Path]:
        model_path     = self.cfg.paths.model_dir / "best_model.keras"
        tokenizer_path = self.cfg.paths.model_dir / "tokenizer.pkl"
        model.save(str(model_path))
        log.info("Model saved -> %s", model_path)
        data.tokenizer.save(tokenizer_path)
        log.info("Tokenizer saved -> %s", tokenizer_path)
        return {"model": model_path, "tokenizer": tokenizer_path}

    def _flat_params(self) -> dict[str, object]:
        cfg = self.cfg
        return {
            "vocab_size":      cfg.data.vocab_size,
            "max_len":         cfg.data.max_sequence_len,
            "model_name":      cfg.model.name,
            "embedding_dim":   cfg.model.embedding_dim,
            "gru_units":       str(cfg.model.gru_units),
            "attention":       cfg.model.attention,
            "batch_size":      cfg.training.batch_size,
            "epochs":          cfg.training.epochs,
            "learning_rate":   cfg.training.learning_rate,
            "label_smoothing": cfg.training.label_smoothing,
            "dropout_rate":    cfg.model.dropout_rate,
        }

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.cfg.logging.level.upper(), logging.INFO),
            format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def _setup_mixed_precision(self) -> None:
        if self.cfg.training.mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            log.info("Mixed precision enabled (float16).")
