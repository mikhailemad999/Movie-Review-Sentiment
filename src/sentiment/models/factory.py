"""
factory.py — Model factory with registry pattern.
"""

from __future__ import annotations

from typing import Callable, ClassVar

from sentiment.config import ModelConfig
from sentiment.exceptions import ModelError
from sentiment.models.base import BaseSentimentModel
from sentiment.models.gru import GRUSentimentModel

ModelConstructor = Callable[[ModelConfig], BaseSentimentModel]


class ModelRegistry:
    _registry: ClassVar[dict[str, ModelConstructor]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[ModelConstructor], ModelConstructor]:
        def decorator(fn: ModelConstructor) -> ModelConstructor:
            cls._registry[name] = fn
            return fn
        return decorator

    @classmethod
    def create(cls, cfg: ModelConfig) -> BaseSentimentModel:
        if cfg.name not in cls._registry:
            raise ModelError(
                f"Unknown model '{cfg.name}'",
                context={"available": list(cls._registry)},
            )
        return cls._registry[cfg.name](cfg)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry)


@ModelRegistry.register("gru_sentiment_v1")
def _build_gru_v1(cfg: ModelConfig) -> GRUSentimentModel:
    return GRUSentimentModel(cfg)
