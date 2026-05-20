"""
predictor.py — Thread-safe, cached inference engine.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from sentiment.config import APISettings
from sentiment.data.preprocessing import SentimentTokenizer
from sentiment.exceptions import ModelLoadError, ModelNotFoundError, ServingError
from sentiment.models.gru import BahdanauAttention
from sentiment.serving.schemas import BatchSentimentResult, SentimentResult

log = logging.getLogger(__name__)


class Predictor:
    """
    Stateful inference engine. Thread-safe; loads model once on construction.

    Example::

        predictor = Predictor.from_settings(APISettings())
        result    = predictor.predict("This movie was fantastic!")
    """

    _lock = threading.Lock()

    def __init__(
        self,
        model:     tf.keras.Model,
        tokenizer: SentimentTokenizer,
        threshold: float = 0.5,
    ) -> None:
        self._model     = model
        self._tokenizer = tokenizer
        self.threshold  = threshold
        log.info("Predictor ready — threshold=%.2f", threshold)

    @classmethod
    def from_paths(
        cls,
        model_path:     str | Path,
        tokenizer_path: str | Path,
        threshold:      float = 0.5,
    ) -> "Predictor":
        model_path     = Path(model_path)
        tokenizer_path = Path(tokenizer_path)

        if not model_path.exists():
            raise ModelNotFoundError(
                "Saved model not found — run train.py first",
                context={"path": str(model_path)},
            )
        try:
            model = tf.keras.models.load_model(
                str(model_path),
                compile=False,
                custom_objects={"BahdanauAttention": BahdanauAttention},
            )
        except Exception as exc:
            raise ModelLoadError("Failed to deserialise model") from exc

        tokenizer = SentimentTokenizer.load(tokenizer_path)
        return cls(model, tokenizer, threshold)

    @classmethod
    def from_settings(cls, settings: APISettings) -> "Predictor":
        return cls.from_paths(settings.model_path, settings.tokenizer_path)

    def predict(self, text: str) -> SentimentResult:
        t0   = time.perf_counter()
        prob = self._predict_batch([text])[0]
        ms   = (time.perf_counter() - t0) * 1000
        return self._build_result(text, prob, ms)

    def predict_batch(self, texts: list[str]) -> BatchSentimentResult:
        t0    = time.perf_counter()
        probs = self._predict_batch(texts)
        total_ms = (time.perf_counter() - t0) * 1000

        results = [self._build_result(t, p) for t, p in zip(texts, probs)]
        return BatchSentimentResult(
            results=results,
            total=len(results),
            n_positive=sum(1 for r in results if r.is_positive),
            n_negative=sum(1 for r in results if not r.is_positive),
            avg_confidence=float(np.mean([r.confidence for r in results])),
            latency_ms=round(total_ms, 2),
        )

    def _predict_batch(self, texts: list[str]) -> np.ndarray:
        try:
            with self._lock:
                X = self._tokenizer.transform(texts)
                return self._model.predict(X, verbose=0).flatten()
        except Exception as exc:
            raise ServingError("Inference failed") from exc

    def _build_result(self, text: str, prob: float, latency_ms: float | None = None) -> SentimentResult:
        is_pos = bool(prob >= self.threshold)
        conf   = float(prob if is_pos else 1 - prob)
        return SentimentResult(
            text=text,
            label="positive" if is_pos else "negative",
            probability=float(prob),
            confidence=conf,
            is_positive=is_pos,
            latency_ms=round(latency_ms, 2) if latency_ms else None,
        )
