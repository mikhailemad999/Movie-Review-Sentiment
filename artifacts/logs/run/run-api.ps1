Set-Location "E:\AMIT AI\Movie Review Sentiment"
python -m uvicorn sentiment.serving.api:create_app --factory --app-dir src --host 127.0.0.1 --port 8000 *> "artifacts\logs\run\api.combined.log"
