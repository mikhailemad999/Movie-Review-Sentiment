"""
schemas.py — Pydantic v2 request/response models for the REST API.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator

MAX_REVIEW_LEN = 5_000
MIN_REVIEW_LEN = 5


class ReviewRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=MIN_REVIEW_LEN,
        max_length=MAX_REVIEW_LEN,
        description="Raw movie review text.",
    )

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Review text must not be blank.")
        return v.strip()


class BatchReviewRequest(BaseModel):
    reviews: list[ReviewRequest] = Field(..., min_length=1, max_length=64)


class SentimentResult(BaseModel):
    text:        str
    label:       str
    probability: float = Field(ge=0.0, le=1.0)
    confidence:  float = Field(ge=0.5, le=1.0)
    is_positive: bool
    latency_ms:  float | None = None


class BatchSentimentResult(BaseModel):
    results:        list[SentimentResult]
    total:          int
    n_positive:     int
    n_negative:     int
    avg_confidence: float
    latency_ms:     float


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    version:      str


class ModelInfoResponse(BaseModel):
    model_name: str
    vocab_size: int
    max_len:    int
    n_params:   int
    framework:  str = "TensorFlow/Keras"
