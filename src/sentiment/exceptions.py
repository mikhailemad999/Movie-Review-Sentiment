"""
exceptions.py — Custom exception hierarchy.
"""

from __future__ import annotations


class SentimentBaseError(Exception):
    def __init__(self, message: str, *, context: dict | None = None) -> None:
        super().__init__(message)
        self.context = context or {}

    def __str__(self) -> str:
        base = super().__str__()
        if self.context:
            ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{base} [{ctx}]"
        return base


class DataError(SentimentBaseError):
    """Dataset loading / preprocessing failures."""

class TokenizerNotFittedError(DataError):
    """predict() called before fit()."""

class ModelError(SentimentBaseError):
    """Model construction or forward-pass failures."""

class ModelNotFoundError(ModelError):
    """Saved model artifact cannot be located."""

class ModelLoadError(ModelError):
    """Deserialization of a saved model failed."""

class TrainingError(SentimentBaseError):
    """Training-loop failures."""

class CheckpointError(TrainingError):
    """Saving / loading a checkpoint failed."""

class ServingError(SentimentBaseError):
    """Inference-pipeline failures."""

class InvalidInputError(ServingError):
    """API received malformed or out-of-range input."""
