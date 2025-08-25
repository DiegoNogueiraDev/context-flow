# Codebase Structure

## Directory Layout
```
rag-mvp/
├── src/
│   ├── core/               # Core business logic
│   │   ├── embedding_service.py    # EmbeddingService class
│   │   ├── text_processor.py       # TextProcessor class  
│   │   ├── rag_service.py          # RAGService class (main orchestrator)
│   │   └── models.py               # Data models
│   ├── storage/            # Data persistence
│   │   └── vector_store.py         # VectorStore class
│   ├── api/               # FastAPI application
│   │   └── main.py                 # API endpoints and app setup
│   └── web/               # Frontend assets
│       ├── templates/index.html    # Main UI template
│       └── static/                 # Static assets
├── tests/                 # Test suite
│   ├── test_embedding_service.py
│   ├── test_text_processor.py
│   ├── test_rag_service.py
│   └── test_vector_store.py
├── requirements.txt       # Python dependencies
├── pytest.ini           # pytest configuration
└── run.py               # Application entry point
```

## Architecture Pattern
- **Layered architecture**: Core → Storage → API
- **Service-oriented**: Each component has a specific responsibility
- **Dependency injection**: Services are injected into the API layer
- **Separation of concerns**: Business logic separate from web layer