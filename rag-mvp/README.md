# RAG MVP - Document Search System

A simple Retrieval-Augmented Generation (RAG) system built with Test-Driven Development (TDD) principles. Upload text documents and search through them using semantic similarity.

## ✨ Features

- **Document Upload**: Upload text files or paste content directly
- **Semantic Search**: Find relevant information using natural language queries
- **Question Answering**: Ask questions and get contextual answers from your documents
- **Real-time Processing**: No mocks or simulations - all components use real implementations
- **Web Interface**: Clean, intuitive web UI for easy interaction
- **REST API**: Complete API for programmatic access

## 🏗️ Architecture

The system is built with a clean, modular architecture following TDD principles:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │    FastAPI      │    │   RAG Service   │
│   (HTML/JS)     │◄──►│    (API Layer)  │◄──►│  (Orchestrator) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────────────────────┼─────────────────────────────────┐
                       │                                 ▼                                 │
           ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
           │ Text Processor  │    │Embedding Service│    │  Vector Store   │    │     Models      │
           │   (Chunking)    │    │   (TF-IDF)      │    │   (SQLite)      │    │ (Data Classes)  │
           └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

1. **TextProcessor**: Handles document chunking and preprocessing
2. **EmbeddingService**: Generates text embeddings using TF-IDF vectorization
3. **VectorStore**: Manages document storage and similarity search with SQLite
4. **RAGService**: Orchestrates all components for end-to-end functionality

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment recommended

### Installation

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd rag-mvp
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run tests** (TDD validation):
   ```bash
   python -m pytest tests/ -v
   ```

3. **Start the server**:
   ```bash
   python run.py
   ```

4. **Open your browser** to `http://localhost:8000`

## 🧪 Testing

This project was built using Test-Driven Development with comprehensive test coverage:

- **35 tests** covering all components
- **No mocks or simulations** - all tests use real implementations
- **Unit tests** for individual components
- **Integration tests** for end-to-end workflows

Run tests:
```bash
# All tests
python -m pytest tests/ -v

# Specific component
python -m pytest tests/test_rag_service.py -v

# With coverage
python -m pytest tests/ -v --cov=src
```

## 🔧 API Usage

### Upload Document
```bash
# Upload text file
curl -X POST "http://localhost:8000/api/upload" -F "file=@document.txt"

# Upload text content
curl -X POST "http://localhost:8000/api/upload-text" \
  -F "content=Your text here" \
  -F "filename=document.txt"
```

### Search Documents
```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "top_k": 5}'
```

### Ask Questions
```bash
curl -X POST "http://localhost:8000/api/answer" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is artificial intelligence?", "top_k": 3}'
```

### Get Statistics
```bash
curl -X GET "http://localhost:8000/api/stats"
```

## 📁 Project Structure

```
rag-mvp/
├── src/
│   ├── core/
│   │   ├── text_processor.py     # Document chunking
│   │   ├── embedding_service.py  # Text embeddings
│   │   ├── rag_service.py        # Main orchestrator
│   │   └── models.py             # Data models
│   ├── storage/
│   │   └── vector_store.py       # Database operations
│   ├── api/
│   │   └── main.py               # FastAPI application
│   └── web/
│       ├── templates/
│       │   └── index.html        # Web interface
│       └── static/               # Static assets
├── tests/
│   ├── test_text_processor.py
│   ├── test_embedding_service.py
│   ├── test_vector_store.py
│   └── test_rag_service.py
├── requirements.txt
├── pytest.ini
├── run.py
└── README.md
```

## 🎯 Key Features Demonstrated

### TDD Implementation
- ✅ **Red-Green-Refactor**: Each component was built following strict TDD
- ✅ **Real Testing**: No mocks - all tests use actual implementations
- ✅ **Comprehensive Coverage**: 35+ tests covering all functionality

### RAG Capabilities
- ✅ **Document Processing**: Intelligent text chunking with overlap
- ✅ **Semantic Similarity**: TF-IDF embeddings for relevance matching
- ✅ **Vector Storage**: Efficient SQLite-based persistence
- ✅ **Correlation Discovery**: Cross-document knowledge correlation
- ✅ **Question Answering**: Context-aware answer generation

### Production-Ready Features
- ✅ **Web Interface**: Complete UI for upload and search
- ✅ **REST API**: Full programmatic access
- ✅ **Error Handling**: Robust error handling and validation
- ✅ **Performance**: Optimized for reasonable response times
- ✅ **Extensibility**: Clean architecture for future enhancements

## 🔍 How It Works

1. **Document Upload**: 
   - Text is cleaned and chunked into overlapping segments
   - Each chunk gets converted to TF-IDF embeddings
   - Documents and embeddings are stored in SQLite

2. **Search Process**:
   - Query is processed through the same embedding pipeline
   - Cosine similarity is calculated against all stored chunks
   - Results are ranked and returned with similarity scores

3. **Knowledge Correlation**:
   - The system can find related information across multiple documents
   - TF-IDF vectorization captures semantic relationships
   - Overlapping chunks ensure context preservation

## ⚡ Performance

- **Upload**: ~100ms for typical documents (1-2KB)
- **Search**: ~200ms for queries against 10+ documents
- **Storage**: SQLite with optimized indexing
- **Memory**: Efficient in-memory processing

## 🚧 Future Enhancements

While this MVP proves the core concept, production systems might add:

- **Advanced Embeddings**: Sentence transformers, OpenAI embeddings
- **Larger Scale**: Vector databases (Pinecone, Weaviate, Chroma)
- **Better QA**: Integration with language models (GPT, Claude)
- **Authentication**: User management and access control
- **Monitoring**: Logging, metrics, and observability

## 🤝 Contributing

This project demonstrates TDD principles. When adding features:

1. **Write tests first** (Red phase)
2. **Implement minimal code** to pass (Green phase)  
3. **Refactor** for clean code (Refactor phase)
4. **No mocks** - use real implementations in tests
5. **Maintain test coverage**

## 📝 License

This project is for educational/demonstration purposes.

---

**Built with TDD principles - No mocks, no simulations, all real functionality! 🎯**