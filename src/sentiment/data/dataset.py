"""
dataset.py — High-performance tf.data input pipeline.

Features:
  * Parallel preprocessing with tf.data
  * Caching and prefetching
  * Stratified train/val split
  * Class-weight computation for imbalanced datasets
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.datasets import imdb

from sentiment.config import DataConfig
from sentiment.data.preprocessing import SentimentTokenizer
from sentiment.exceptions import DataError

log     = logging.getLogger(__name__)
AUTOTUNE = tf.data.AUTOTUNE


@dataclass
class Split:
    dataset:   tf.data.Dataset
    size:      int
    pos_ratio: float


@dataclass
class IMDBData:
    train:         Split
    val:           Split
    test:          Split
    tokenizer:     SentimentTokenizer
    class_weights: dict[int, float]


class IMDBDataLoader:
    """Downloads IMDB, decodes to text, tokenises, and returns tf.data pipelines."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg

    def load(self, batch_size: int = 64) -> IMDBData:
        log.info("Loading IMDB (num_words=%d) ...", self.cfg.vocab_size)
        X_train_raw, X_test_raw, y_train_all, y_test = self._download()

        tokenizer   = SentimentTokenizer(self.cfg.vocab_size, self.cfg.max_sequence_len)
        X_train_seq = tokenizer.fit_transform(X_train_raw)
        X_test_seq  = tokenizer.transform(X_test_raw)

        y_train_all = np.array(y_train_all, dtype=np.float32)
        y_test      = np.array(y_test,      dtype=np.float32)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_seq, y_train_all,
            test_size=self.cfg.val_split,
            stratify=y_train_all,
            random_state=self.cfg.random_seed,
        )

        class_weights = self._compute_class_weights(y_tr)

        return IMDBData(
            train=self._make_split(X_tr,  y_tr,  batch_size, shuffle=True),
            val=self._make_split(X_val, y_val, batch_size),
            test=self._make_split(X_test_seq, y_test, batch_size),
            tokenizer=tokenizer,
            class_weights=class_weights,
        )

    def _download(self) -> tuple[list[str], list[str], list[int], list[int]]:
        try:
            (X_tr_int, y_tr), (X_te_int, y_te) = imdb.load_data(num_words=self.cfg.vocab_size)
        except Exception as exc:
            raise DataError("Failed to download IMDB dataset") from exc

        idx2word = {v + 3: k for k, v in imdb.get_word_index().items()}
        idx2word |= {0: "<PAD>", 1: "<START>", 2: "<UNK>", 3: "<UNUSED>"}

        def decode(seq: np.ndarray) -> str:
            return " ".join(idx2word.get(i, "<UNK>") for i in seq)

        return (
            [decode(s) for s in X_tr_int],
            [decode(s) for s in X_te_int],
            list(y_tr), list(y_te),
        )

    @staticmethod
    def _make_split(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = False) -> Split:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=False).cache().prefetch(AUTOTUNE)
        return Split(dataset=ds, size=len(X), pos_ratio=float(y.mean()))

    @staticmethod
    def _compute_class_weights(y: np.ndarray) -> dict[int, float]:
        w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
        return {0: float(w[0]), 1: float(w[1])}
