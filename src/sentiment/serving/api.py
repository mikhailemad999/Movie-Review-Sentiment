"""
api.py — Production FastAPI REST service.

Endpoints:
    GET  /health          - liveness probe
    GET  /model/info      - model metadata
    POST /predict         - single review
    POST /predict/batch   - batch reviews (max 64)

Extras: Prometheus metrics, structured logging, async-safe design.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from sentiment.config import APISettings
from sentiment.exceptions import InvalidInputError, ModelNotFoundError, ServingError
from sentiment.serving.predictor import Predictor
from sentiment.serving.schemas import (
    BatchReviewRequest, BatchSentimentResult,
    HealthResponse, ModelInfoResponse,
    ReviewRequest, SentimentResult,
)

log = logging.getLogger(__name__)

REQUEST_COUNT   = Counter("sentiment_requests_total", "Total prediction requests", ["endpoint", "label"])
REQUEST_LATENCY = Histogram("sentiment_request_latency_seconds", "Prediction latency", ["endpoint"])


predictor_store: dict[str, Predictor] = {}

def get_predictor() -> Predictor:
    p = predictor_store.get("main")
    if p is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded.")
    return p

def create_app(settings: APISettings | None = None) -> FastAPI:
    settings = settings or APISettings()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        log.info("Loading model from %s ...", settings.model_path)
        try:
            predictor_store["main"] = Predictor.from_settings(settings)
            log.info("Predictor ready.")
        except ModelNotFoundError:
            log.critical("Model not found — start server after training!")
        yield
        predictor_store.clear()

    app = FastAPI(
        title="GRU Sentiment Analysis API",
        description="Bidirectional GRU sentiment classifier for IMDB reviews.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def log_latency(request: Request, call_next):  # type: ignore[no-untyped-def]
        t0       = time.perf_counter()
        response = await call_next(request)
        ms       = (time.perf_counter() - t0) * 1000
        log.info("%s %s - %dms", request.method, request.url.path, round(ms))
        return response

    @app.exception_handler(ServingError)
    async def serving_error_handler(request: Request, exc: ServingError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"error": str(exc), "context": exc.context})

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", model_loaded="main" in predictor_store, version="1.0.0")

    @app.get("/metrics", tags=["System"])
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
    async def model_info(predictor: Predictor = Depends(get_predictor)) -> ModelInfoResponse:
        return ModelInfoResponse(
            model_name=predictor._tokenizer.__class__.__name__,
            vocab_size=predictor._tokenizer.vocab_size,
            max_len=predictor._tokenizer.max_len,
            n_params=int(predictor._model.count_params()),
        )

    @app.post("/predict", response_model=SentimentResult, tags=["Prediction"])
    async def predict(body: ReviewRequest, predictor: Predictor = Depends(get_predictor)) -> SentimentResult:
        with REQUEST_LATENCY.labels(endpoint="/predict").time():
            result = predictor.predict(body.text)
        REQUEST_COUNT.labels(endpoint="/predict", label=result.label).inc()
        return result

    @app.post("/predict/batch", response_model=BatchSentimentResult, tags=["Prediction"])
    async def predict_batch(body: BatchReviewRequest, predictor: Predictor = Depends(get_predictor)) -> BatchSentimentResult:
        texts  = [r.text for r in body.reviews]
        with REQUEST_LATENCY.labels(endpoint="/predict/batch").time():
            result = predictor.predict_batch(texts)
        for r in result.results:
            REQUEST_COUNT.labels(endpoint="/predict/batch", label=r.label).inc()
        return result

    return app


def start() -> None:
    settings = APISettings()
    uvicorn.run("sentiment.serving.api:create_app", factory=True,
                host=settings.host, port=settings.port,
                workers=settings.workers, log_level=settings.log_level)

if __name__ == "__main__":
    start()
