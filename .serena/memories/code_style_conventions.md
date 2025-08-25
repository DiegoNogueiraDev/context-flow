# Code Style and Conventions

## Python Style
- **PEP 8 compliance**: Standard Python style guide
- **Type hints**: Used throughout for better code clarity
- **Class names**: PascalCase (e.g., `EmbeddingService`, `RAGService`)
- **Function/method names**: snake_case (e.g., `process_text`, `search_documents`)  
- **Variable names**: snake_case (e.g., `chunk_size`, `embedding_dim`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_CHUNK_SIZE`)

## Project Conventions
- **Service pattern**: Each core component is a service class
- **Dependency injection**: Services passed to constructors
- **Error handling**: Proper exception handling with meaningful messages
- **Async/await**: Used for I/O operations in FastAPI endpoints
- **Path handling**: Use `pathlib.Path` for file operations

## FastAPI Conventions
- **Response models**: Pydantic models for API responses
- **Request validation**: Pydantic models for request bodies
- **Endpoint naming**: RESTful conventions where applicable
- **Status codes**: Proper HTTP status codes (200, 201, 404, 422, etc.)

## Testing Conventions
- **Test file naming**: `test_*.py`
- **Test class naming**: `Test*` (optional, mostly using functions)
- **Test function naming**: `test_*`
- **Assertions**: Use pytest assertions (`assert`)
- **Fixtures**: Use pytest fixtures for setup/teardown
- **No mocks**: Real integration testing philosophy

## Documentation
- **Docstrings**: Triple-quoted strings for classes and methods
- **Comments**: Inline comments for complex logic
- **Type annotations**: Self-documenting code through types