"""Integration tests for the FastAPI service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sentiment.serving.api import create_app
from sentiment.serving.predictor import Predictor


@pytest.fixture
def client(predictor: Predictor) -> TestClient:
    app = create_app()

    from sentiment.serving import api as api_module

    app.dependency_overrides[api_module.get_predictor] = lambda: predictor  # type: ignore[attr-defined]
    return TestClient(app)


class TestHealthEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/health").status_code == 200


class TestPredictEndpoint:
    def test_valid_review(self, client: TestClient) -> None:
        r = client.post("/predict", json={"text": "This movie was absolutely fantastic!"})
        assert r.status_code == 200
        body = r.json()
        assert body["label"] in {"positive", "negative"}
        assert 0.0 <= body["probability"] <= 1.0
        assert body["confidence"] >= 0.5

    def test_blank_text_rejected(self, client: TestClient) -> None:
        assert client.post("/predict", json={"text": "   "}).status_code == 422

    def test_too_short_rejected(self, client: TestClient) -> None:
        assert client.post("/predict", json={"text": "ok"}).status_code == 422

    def test_batch_predict(self, client: TestClient) -> None:
        payload = {"reviews": [
            {"text": "Incredible masterpiece, deeply moving."},
            {"text": "Boring and predictable. Complete disappointment."},
        ]}
        r    = client.post("/predict/batch", json=payload)
        body = r.json()
        assert r.status_code == 200
        assert body["total"] == 2
        assert body["n_positive"] + body["n_negative"] == 2

    def test_batch_over_limit_rejected(self, client: TestClient) -> None:
        payload = {"reviews": [{"text": "Some review text here."}] * 65}
        assert client.post("/predict/batch", json=payload).status_code == 422
