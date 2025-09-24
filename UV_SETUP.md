# Code Indexer - UV Package Manager Setup

# Install dependencies using UV
uv pip install -r requirements.txt

# For development mode with auto-reload
uv run uvicorn api.rest:app --reload --host 127.0.0.1 --port 8000

# For production
uv run uvicorn api.rest:app --host 127.0.0.1 --port 8000 --workers 4