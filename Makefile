.PHONY: install train evaluate serve app test lint type-check docker-up docker-down clean

install:
	pip install -e ".[dev]"
	pre-commit install

train:
	python scripts/train.py --config configs/default.yaml

train-large:
	python scripts/train.py --config configs/experiment_gru_large.yaml

evaluate:
	python scripts/evaluate.py

serve:
	uvicorn "sentiment.serving.api:create_app" --factory \
	  --host 0.0.0.0 --port 8000 --reload

app:
	streamlit run app/streamlit_app.py

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

test:
	pytest tests/ -v --cov=src/sentiment --cov-report=term-missing

lint:
	ruff check src/ scripts/ tests/
	ruff format --check src/ scripts/ tests/

type-check:
	mypy src/sentiment --ignore-missing-imports

docker-up:
	docker compose -f docker/docker-compose.yml up --build -d

docker-down:
	docker compose -f docker/docker-compose.yml down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
