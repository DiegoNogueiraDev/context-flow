from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.rag_service import RAGService


app = FastAPI(
    title="RAG MVP",
    description="Simple RAG system for text upload and semantic search",
    version="1.0.0"
)

# Static files and templates
static_dir = os.path.join(os.path.dirname(__file__), '..', 'web', 'static')
template_dir = os.path.join(os.path.dirname(__file__), '..', 'web', 'templates')

# Create directories if they don't exist
os.makedirs(static_dir, exist_ok=True)
os.makedirs(template_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=template_dir)

# Initialize RAG service
rag_service = RAGService("rag_database.db")


# Pydantic models for API
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    mode: Optional[str] = "semantic"


class SearchResult(BaseModel):
    content: str
    similarity: float
    filename: str
    document_id: str
    chunk_id: str


class DocumentResponse(BaseModel):
    id: str
    filename: str
    content: str
    chunks: List[Dict[str, Any]]


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    message: str
    chunks_count: int


class StatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    average_chunks_per_document: float


# Web interface routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main web interface"""
    return templates.TemplateResponse("index.html", {"request": request})


# API routes
@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a text document for processing"""
    try:
        # Read file content
        content = await file.read()
        
        # Process document
        document_id = rag_service.upload_document_bytes(content, file.filename)
        
        # Get document details for response
        doc_details = rag_service.get_document_details(document_id)
        chunks_count = len(doc_details['chunks']) if doc_details else 0
        
        return UploadResponse(
            document_id=document_id,
            filename=file.filename,
            message=f"Document '{file.filename}' uploaded and processed successfully",
            chunks_count=chunks_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/api/upload-text", response_model=UploadResponse)
async def upload_text_document(content: str = Form(...), filename: str = Form(...)):
    """Upload text content directly"""
    try:
        document_id = rag_service.upload_document(content, filename)
        
        # Get document details for response
        doc_details = rag_service.get_document_details(document_id)
        chunks_count = len(doc_details['chunks']) if doc_details else 0
        
        return UploadResponse(
            document_id=document_id,
            filename=filename,
            message=f"Text content uploaded as '{filename}' successfully",
            chunks_count=chunks_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")


@app.post("/api/search", response_model=List[SearchResult])
async def search_documents(search_request: SearchRequest):
    """Search for similar content in uploaded documents"""
    try:
        if not search_request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        results = rag_service.search(
            query=search_request.query,
            top_k=search_request.top_k,
            mode=search_request.mode
        )
        
        return [SearchResult(**result) for result in results]
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/api/documents", response_model=List[Dict[str, Any]])
async def get_all_documents():
    """Get all uploaded documents"""
    try:
        return rag_service.get_all_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")


@app.get("/api/documents/{document_id}", response_model=DocumentResponse)
async def get_document_details(document_id: str):
    """Get detailed information about a specific document"""
    try:
        doc_details = rag_service.get_document_details(document_id)
        if not doc_details:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentResponse(**doc_details)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its chunks"""
    try:
        success = rag_service.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": "Document deleted successfully", "document_id": document_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


@app.post("/api/answer")
async def answer_question(search_request: SearchRequest):
    """Answer a question based on uploaded documents"""
    try:
        if not search_request.query.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        result = rag_service.answer_question(
            question=search_request.query,
            max_context_chunks=search_request.top_k or 3
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")


@app.get("/api/stats", response_model=StatsResponse)
async def get_statistics():
    """Get knowledge base statistics"""
    try:
        stats = rag_service.get_statistics()
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "RAG MVP is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)