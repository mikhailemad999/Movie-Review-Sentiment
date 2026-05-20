Set-Location "E:\AMIT AI\Movie Review Sentiment"
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true *> "artifacts\logs\run\streamlit.combined.log"
