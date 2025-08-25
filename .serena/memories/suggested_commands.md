# Suggested Commands for RAG MVP Development

## Development Environment Setup
```bash
# Navigate to project
cd rag-mvp

# Create virtual environment (if needed)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the Application
```bash
# Development server with auto-reload
python run.py

# Alternative using uvicorn directly
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## Testing Commands
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_embedding_service.py

# Run with coverage (if installed)
pytest --cov=src

# Quick test run (quiet mode)  
pytest -q
```

## Environment Variables
```bash
# Embedding model configuration
export EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export MODEL_LOCAL_PATH="/path/to/local/model"  # Optional
export EMBEDDING_BACKEND="st"  # or "tfidf" to force backend
```

## Manual Testing
```bash
# Upload test file
curl -F "file=@sample_data/sample.txt" http://localhost:8000/upload

# Search documents  
curl "http://localhost:8000/search?q=semantic%20search&k=5"

# Check health
curl http://localhost:8000/healthz

# Get statistics
curl http://localhost:8000/stats
```