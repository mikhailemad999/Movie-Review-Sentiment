"""
gru.py — Bidirectional GRU with additive (Bahdanau) self-attention.

Architecture:
    Embedding -> SpatialDropout1D
    -> N x Bidirectional GRU (stacked, return_sequences=True)
    -> BahdanauAttention (optional)
    -> GlobalAvgPool + GlobalMaxPool -> Concat
    -> M x Dense (ReLU, L2, BatchNorm, Dropout)
    -> Dense(1, sigmoid)
"""

from __future__ import annotations

import logging
import tensorflow as tf
from tensorflow.keras import Input, Model, layers, regularizers

from sentiment.config import ModelConfig, TrainingConfig
from sentiment.models.base import BaseSentimentModel

log = logging.getLogger(__name__)


@tf.keras.utils.register_keras_serializable()
class BahdanauAttention(layers.Layer):
    """
    Soft self-attention over the sequence axis.
    Returns a context vector weighted by learned alignment scores.
    """

    def __init__(self, units: int = 64, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.W = layers.Dense(units, use_bias=False)
        self.V = layers.Dense(1,     use_bias=False)

    def call(self, H: tf.Tensor, training: bool = False) -> tf.Tensor:
        # H: (batch, seq_len, hidden)
        score   = self.V(tf.nn.tanh(self.W(H)))       # (batch, seq_len, 1)
        alpha   = tf.nn.softmax(score, axis=1)         # (batch, seq_len, 1)
        context = tf.reduce_sum(alpha * H, axis=1)     # (batch, hidden)
        return context

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"units": self.W.units})
        return config


class GRUSentimentModel(BaseSentimentModel):
    """Production-grade Bidirectional GRU sentiment classifier."""

    def build(
        self,
        vocab_size:   int,
        max_len:      int,
        training_cfg: TrainingConfig | None = None,
    ) -> Model:
        cfg   = self.cfg
        t_cfg = training_cfg or TrainingConfig()
        reg   = regularizers.l2(cfg.l2_reg)

        inp = Input(shape=(max_len,), name="token_ids")

        x = layers.Embedding(vocab_size, cfg.embedding_dim, mask_zero=True, name="embedding")(inp)
        x = layers.SpatialDropout1D(cfg.spatial_dropout, name="spatial_drop")(x)

        for i, units in enumerate(cfg.gru_units):
            x = layers.Bidirectional(
                layers.GRU(
                    units,
                    return_sequences=True,
                    dropout=cfg.dropout_rate * 0.5,
                    recurrent_dropout=0.0,   # keep cuDNN-compatible
                    name=f"gru_{i}",
                ),
                name=f"bi_gru_{i}",
            )(x)

        if cfg.attention:
            attn = BahdanauAttention(units=cfg.gru_units[-1] * 2, name="attention")(x)
            avg  = layers.GlobalAveragePooling1D(name="avg_pool")(x)
            max_ = layers.GlobalMaxPooling1D(name="max_pool")(x)
            x    = layers.Concatenate(name="pool_concat")([attn, avg, max_])
        else:
            avg  = layers.GlobalAveragePooling1D(name="avg_pool")(x)
            max_ = layers.GlobalMaxPooling1D(name="max_pool")(x)
            x    = layers.Concatenate(name="pool_concat")([avg, max_])

        for j, units in enumerate(cfg.dense_units):
            x = layers.Dense(units, activation="relu", kernel_regularizer=reg, name=f"dense_{j}")(x)
            x = layers.BatchNormalization(name=f"bn_{j}")(x)
            x = layers.Dropout(cfg.dropout_rate, name=f"drop_{j}")(x)

        out   = layers.Dense(1, activation="sigmoid", name="sentiment")(x)
        model = Model(inputs=inp, outputs=out, name="GRUSentiment")

        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=t_cfg.learning_rate,
                weight_decay=t_cfg.weight_decay,
            ),
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=t_cfg.label_smoothing),
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc",    curve="ROC"),
                tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        self._model = model
        log.info("Model built — parameters: %s", f"{model.count_params():,}")
        return model
