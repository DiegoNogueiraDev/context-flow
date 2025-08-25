# RAG MVP Project Overview

## Purpose
A simple and objective RAG (Retrieval Augmented Generation) system that allows users to:
- Upload .txt files
- Perform chunking + vectorization + indexing (embeddings or TF-IDF fallback)
- Search through documents with semantic search
- View correlations between related text chunks

## Key Characteristics
- **TDD approach**: No mocks, tests exercise real endpoints
- **Simple MVP**: Focused on core RAG functionality
- **Fallback system**: sentence-transformers → TF-IDF if not available
- **Real-time indexing**: Immediate processing on upload
- **Correlation feature**: Shows related chunks for each search result

## Tech Stack
- **Backend**: FastAPI + SQLite for persistence + NumPy
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2) with TF-IDF fallback
- **Frontend**: Simple HTML + JavaScript (served by FastAPI)
- **Testing**: pytest + httpx + fastapi.testclient
- **Index**: In-memory cosine similarity, optional FAISS integration