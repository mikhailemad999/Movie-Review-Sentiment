# Movie Review Sentiment

Production-style movie review sentiment analysis project using a TensorFlow/Keras Bidirectional GRU model, FastAPI for serving, and Streamlit for the UI.

## Features

- Train a GRU sentiment classifier on the IMDB dataset.
- Serve predictions through a FastAPI REST API.
- Use a Streamlit dashboard to analyze movie reviews.
- Track training runs with MLflow.
- Run locally with Python or with Docker Compose.

## Project Structure

```text
app/                  Streamlit application
configs/              YAML configuration files
docker/               Dockerfiles and Docker Compose file
scripts/              Training entry points
src/sentiment/        Python package source code
tests/                Unit and integration tests
artifacts/models/     Saved model and tokenizer
mlruns/               MLflow tracking data
```

## Requirements

- Python 3.10 or newer
- pip
- Docker Desktop, optional

The project dependencies are defined in `pyproject.toml`.

## Local Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project:

```powershell
pip install -e ".[dev]"
```

If you do not want the development tools, use:

```powershell
pip install -e .
```

## Train The Model

The API expects these files:

```text
artifacts/models/best_model.keras
artifacts/models/tokenizer.pkl
```

If they are missing, train the model:

```powershell
python scripts/train.py --config configs/default.yaml
```

For a quick smoke-test training run:

```powershell
python scripts/train.py --config configs/default.yaml --epochs 1 --batch 128
```

## Run The API

Start the FastAPI server:

```powershell
uvicorn "sentiment.serving.api:create_app" --factory --host 127.0.0.1 --port 8000 --reload
```

Open the health endpoint:

```text
http://127.0.0.1:8000/health
```

The response should include:

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

Test a prediction:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body '{"text":"This movie was wonderful and moving."}'
```

## Run The Streamlit App

In another terminal, start Streamlit:

```powershell
streamlit run app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

By default, the app calls the API at:

```text
http://localhost:8000
```

To point it at a different API URL:

```powershell
$env:SENTIMENT_API_URL="http://127.0.0.1:8000"
streamlit run app/streamlit_app.py
```

## Run With Docker Compose

Build and start the API, Streamlit app, and MLflow:

```powershell
docker compose -f docker/docker-compose.yml up --build
```

Open:

```text
Streamlit: http://localhost:8501
API:       http://localhost:8000/health
MLflow:    http://localhost:5000
```

Stop the Docker services:

```powershell
docker compose -f docker/docker-compose.yml down
```

## Useful Make Commands

```powershell
make train
make serve
make app
make test
make lint
make docker-up
make docker-down
```

On Windows, if `make` is unavailable, run the equivalent commands shown above directly.

## Tests

Run the test suite:

```powershell
pytest tests/ -v
```

Run linting:

```powershell
ruff check src/ scripts/ tests/
```

## API Endpoints

- `GET /health` - service health and model load status
- `GET /model/info` - model metadata
- `POST /predict` - predict one review
- `POST /predict/batch` - predict up to 64 reviews
- `GET /metrics` - Prometheus metrics

## Environment Variables

The API uses the `SENTIMENT_` prefix:

```text
SENTIMENT_HOST
SENTIMENT_PORT
SENTIMENT_MODEL_PATH
SENTIMENT_TOKENIZER_PATH
SENTIMENT_WORKERS
SENTIMENT_LOG_LEVEL
```

The Streamlit app uses:

```text
SENTIMENT_API_URL
```

## Notes

- Native Windows TensorFlow builds may run on CPU only.
- The first model load or first prediction can take a few seconds.
- If `/health` returns `model_loaded: false`, train the model or check the model paths.
