from typing import List, Dict, Any, Optional
import tempfile
import os

from .text_processor import TextProcessor
from .embedding_service import EmbeddingService
from storage.vector_store import VectorStore


class RAGService:
    """Main service that orchestrates all RAG components"""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize RAG service with all components"""
        if db_path is None:
            # Create temporary database if none provided
            fd, db_path = tempfile.mkstemp(suffix='.db')
            os.close(fd)
        
        self.text_processor = TextProcessor()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore(db_path)
        
        # Initialize database
        self.vector_store.initialize_database()
    
    def upload_document(self, content: str, filename: str) -> str:
        """Upload and process a document from text content"""
        # Process document into chunks
        document = self.text_processor.create_document(content, filename)
        document.chunks = self.text_processor.chunk_text(content)
        
        # Generate embeddings for all chunks
        chunk_texts = [chunk.content for chunk in document.chunks]
        embeddings = self.embedding_service.generate_embeddings(chunk_texts)
        
        # Store in vector database
        self.vector_store.store_document(document, embeddings)
        
        return document.id
    
    def upload_document_bytes(self, file_content: bytes, filename: str) -> str:
        """Upload and process a document from bytes (file upload)"""
        document = self.text_processor.process_upload(file_content, filename)
        
        # Generate embeddings for all chunks
        chunk_texts = [chunk.content for chunk in document.chunks]
        embeddings = self.embedding_service.generate_embeddings(chunk_texts)
        
        # Store in vector database
        self.vector_store.store_document(document, embeddings)
        
        return document.id
    
    def search(self, query: str, top_k: int = 5, mode: str = 'semantic') -> List[Dict[str, Any]]:
        """Search for similar content using semantic similarity"""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        query = query.strip()
        
        # Get all documents to build corpus
        all_docs = self.vector_store.get_all_documents()
        if not all_docs:
            return []
        
        # Get all chunks for building unified corpus
        all_chunks = []
        for doc in all_docs:
            chunks = self.vector_store.get_document_chunks(doc['id'])
            for chunk in chunks:
                chunk['document_id'] = doc['id']
                chunk['filename'] = doc['filename']
            all_chunks.extend(chunks)
        
        if not all_chunks:
            return []
        
        # Build corpus with query and all chunk texts
        chunk_texts = [chunk['content'] for chunk in all_chunks]
        query_embedding, chunk_embeddings = self.embedding_service.embed_query_and_corpus(
            query, chunk_texts
        )
        
        # Find most similar chunks
        similarities = []
        for i, chunk_embedding in enumerate(chunk_embeddings):
            similarity = self.embedding_service.cosine_similarity(query_embedding, chunk_embedding)
            
            chunk_info = all_chunks[i].copy()
            chunk_info['similarity'] = similarity
            similarities.append(chunk_info)
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Format results
        results = []
        for sim in similarities[:top_k]:
            results.append({
                'content': sim['content'],
                'similarity': sim['similarity'],
                'document_id': sim['document_id'],
                'filename': sim['filename'],
                'chunk_id': sim['chunk_id']
            })
        
        return results
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all uploaded documents"""
        return self.vector_store.get_all_documents()
    
    def get_document_details(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific document"""
        docs = self.vector_store.get_all_documents()
        doc = next((d for d in docs if d['id'] == document_id), None)
        
        if not doc:
            return None
        
        # Get chunks for this document
        chunks = self.vector_store.get_document_chunks(document_id)
        
        result = dict(doc)
        result['chunks'] = chunks
        return result
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks"""
        return self.vector_store.delete_document(document_id)
    
    def answer_question(self, question: str, max_context_chunks: int = 3) -> Dict[str, Any]:
        """Answer a question based on stored documents (basic implementation)"""
        # For MVP, we'll return the most relevant chunks as context
        # In a full implementation, this would use a language model to generate answers
        
        search_results = self.search(question, top_k=max_context_chunks)
        
        if not search_results:
            return {
                'answer': "I don't have information to answer that question.",
                'confidence': 0.0,
                'sources': []
            }
        
        # Simple answer generation: return the most relevant chunk's content
        # In production, this would be enhanced with a language model
        top_result = search_results[0]
        
        answer = f"Based on the available information: {top_result['content']}"
        
        return {
            'answer': answer,
            'confidence': top_result['similarity'],
            'sources': [{
                'content': result['content'],
                'filename': result['filename'],
                'similarity': result['similarity']
            } for result in search_results]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        documents = self.get_all_documents()
        
        total_chunks = 0
        for doc in documents:
            chunks = self.vector_store.get_document_chunks(doc['id'])
            total_chunks += len(chunks)
        
        return {
            'total_documents': len(documents),
            'total_chunks': total_chunks,
            'average_chunks_per_document': total_chunks / len(documents) if documents else 0
        }
    
    def close(self):
        """Clean up resources"""
        if hasattr(self.vector_store, 'close'):
            self.vector_store.close()