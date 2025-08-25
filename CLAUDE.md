# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a RAG (Retrieval-Augmented Generation) MVP built following **TDD principles with no mocks** - all tests use real implementations. The system allows uploading text documents and performing semantic search with question answering.

### Core Architecture Pattern
- **Layered architecture**: Core → Storage → API → Web
- **Service-oriented**: Each component has specific responsibilities
- **Dependency injection**: Services are injected into the API layer
- **TDD approach**: Comprehensive test coverage without mocks or simulations

### Key Components
- **TextProcessor**: Document chunking and preprocessing
- **EmbeddingService**: TF-IDF vectorization for semantic similarity
- **VectorStore**: SQLite-based document and vector storage
- **RAGService**: Main orchestrator coordinating all components
- **FastAPI**: REST API layer with automatic OpenAPI documentation

## Development Commands

### Environment Setup
```bash
cd rag-mvp
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Running the Application
```bash
# Start development server with auto-reload
python run.py

# Alternative using uvicorn directly
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing (Critical - TDD Workflow)
```bash
# Run all tests (35+ tests, no mocks)
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_rag_service.py -v
pytest tests/test_embedding_service.py -v
pytest tests/test_vector_store.py -v
pytest tests/test_text_processor.py -v

# Quick test run
pytest -q

# Run with coverage (if installed)
pytest --cov=src -v
```

### Manual API Testing
```bash
# Health check
curl http://localhost:8000/health

# Upload document
curl -X POST "http://localhost:8000/api/upload" -F "file=@document.txt"

# Search
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "semantic search", "top_k": 5}'

# Ask questions
curl -X POST "http://localhost:8000/api/answer" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "top_k": 3}'

# Get statistics
curl http://localhost:8000/api/stats
```

## Directory Structure

```
rag-mvp/
├── src/
│   ├── core/                      # Core business logic
│   │   ├── text_processor.py      # TextProcessor class - chunking/preprocessing
│   │   ├── embedding_service.py   # EmbeddingService - TF-IDF vectorization
│   │   ├── rag_service.py         # RAGService - main orchestrator
│   │   └── models.py              # Data models and types
│   ├── storage/                   # Data persistence layer
│   │   └── vector_store.py        # VectorStore - SQLite operations
│   ├── api/                       # FastAPI application
│   │   └── main.py                # API endpoints and app setup
│   └── web/                       # Frontend assets
│       ├── templates/index.html   # Main UI template
│       └── static/                # Static assets (CSS, JS)
├── tests/                         # Comprehensive test suite
│   ├── test_text_processor.py
│   ├── test_embedding_service.py
│   ├── test_vector_store.py
│   └── test_rag_service.py
├── requirements.txt               # Python dependencies
├── pytest.ini                   # pytest configuration
└── run.py                       # Application entry point
```

## API Endpoints

The FastAPI application (`src/api/main.py`) provides these endpoints:

- `GET /` - Web UI (serves HTML interface)
- `GET /health` - Health check endpoint
- `GET /api/stats` - System statistics (document/chunk counts)
- `POST /api/upload` - Upload text file for indexing
- `POST /api/upload-text` - Upload text content directly  
- `POST /api/search` - Semantic search across documents
- `POST /api/answer` - Question answering with context
- `GET /api/documents` - List all documents
- `GET /api/documents/{doc_id}` - Get document details
- `DELETE /api/documents/{doc_id}` - Delete document
- `GET /docs` - OpenAPI/Swagger documentation

## Key Testing Principles

This codebase follows strict TDD principles:

1. **No mocks or simulations** - all tests use real implementations
2. **Real database operations** - tests create actual SQLite databases
3. **Real embeddings** - tests use actual TF-IDF vectorization
4. **End-to-end testing** - full request/response cycles through FastAPI
5. **Comprehensive coverage** - 35+ tests covering all components

## Development Guidelines

### When Adding Features
1. **Write tests first** (Red phase)
2. **Implement minimal code** to pass (Green phase)  
3. **Refactor** for clean code (Refactor phase)
4. **Never use mocks** - always use real implementations
5. **Maintain modular architecture** - keep components separate

### Code Organization
- **Core logic** goes in `src/core/`
- **Data operations** go in `src/storage/`
- **API endpoints** go in `src/api/main.py`
- **Tests mirror** the source structure in `tests/`

### Performance Characteristics
- **Upload**: ~100ms for typical documents (1-2KB)
- **Search**: ~200ms for queries against 10+ documents  
- **Storage**: SQLite with optimized indexing
- **Memory**: Efficient in-memory TF-IDF processing

## Environment Variables
```bash
# Optional: Override default chunking parameters
export CHUNK_SIZE=500
export CHUNK_OVERLAP=50
```

Access the web UI at `http://localhost:8000` and API documentation at `http://localhost:8000/docs`.

- Desenvolvimento orientado a TDD. Sem uso de mocks ou fluxos simulados, apenas implementações pequenais e bem modularizadas reais.