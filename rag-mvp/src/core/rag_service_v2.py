"""
Enhanced RAG service with support for PDF/Markdown processing and improved scalability.
Maintains backward compatibility with the original RAGService.
"""
from typing import List, Dict, Any, Optional, Union
import tempfile
import os
import logging
import time
from pathlib import Path

from .document_processor import DocumentProcessor, EnhancedDocument
from .embedding_service_v2 import EnhancedEmbeddingService
from ..storage.vector_store_v2 import ChromaVectorStore

# Fallback imports
from .text_processor import TextProcessor
from .embedding_service import EmbeddingService
from ..storage.vector_store import VectorStore


class EnhancedRAGService:
    """Enhanced RAG service with improved document processing and scalability"""
    
    def __init__(self, 
                 db_path: Optional[str] = None,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 use_chromadb: bool = True,
                 enable_caching: bool = True,
                 chroma_persist_dir: str = "chroma_db"):
        """
        Initialize enhanced RAG service
        
        Args:
            db_path: Path for SQLite database (fallback/metadata)
            embedding_model: Name of sentence-transformer model
            use_chromadb: Whether to use ChromaDB for vector storage
            enable_caching: Whether to enable embedding caching
            chroma_persist_dir: Directory for ChromaDB persistence
        """
        self.db_path = db_path or "rag_enhanced.db"
        self.embedding_model = embedding_model
        self.use_chromadb = use_chromadb
        self.enable_caching = enable_caching
        
        # Initialize document processor
        self.document_processor = DocumentProcessor(use_docling=True)
        
        # Fallback to text processor for compatibility
        self.text_processor = TextProcessor()
        
        # Initialize embedding service
        try:
            self.embedding_service = EnhancedEmbeddingService(
                model_name=embedding_model,
                use_cache=enable_caching,
                fallback_to_tfidf=True
            )
            logging.info(f"Enhanced embedding service initialized: {embedding_model}")
        except Exception as e:
            logging.error(f"Failed to initialize enhanced embedding service: {e}")
            logging.info("Falling back to TF-IDF embedding service")
            self.embedding_service = EmbeddingService()
        
        # Initialize vector store
        if use_chromadb:
            try:
                self.vector_store = ChromaVectorStore(
                    persist_directory=chroma_persist_dir,
                    fallback_db_path=self.db_path
                )
                logging.info("ChromaDB vector store initialized")
            except Exception as e:
                logging.error(f"Failed to initialize ChromaDB: {e}")
                logging.info("Falling back to SQLite vector store")
                self.vector_store = VectorStore(self.db_path)
        else:
            self.vector_store = VectorStore(self.db_path)
        
        # Initialize database
        self.vector_store.initialize_database()
        
        # Track processing statistics
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'searches_performed': 0,
            'total_processing_time': 0,
            'average_processing_time': 0
        }
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return self.document_processor.get_supported_formats()
    
    def is_supported_format(self, filename: str) -> bool:
        """Check if file format is supported"""
        return self.document_processor.is_supported_format(filename)
    
    def upload_document_file(self, file_path: str, filename: Optional[str] = None) -> str:
        """Upload and process a document file"""
        start_time = time.time()
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if filename is None:
            filename = os.path.basename(file_path)
        
        if not self.is_supported_format(filename):
            raise ValueError(f"Unsupported file format: {Path(filename).suffix}")
        
        try:
            # Process document
            document = self.document_processor.process_file(file_path, filename)
            
            # Generate embeddings
            chunk_texts = [chunk.content for chunk in document.chunks]
            embeddings = self.embedding_service.generate_embeddings(chunk_texts)
            
            # Store in vector database
            success = self.vector_store.store_document(document, embeddings)
            
            if not success:
                raise RuntimeError("Failed to store document in vector database")
            
            # Update stats
            processing_time = time.time() - start_time
            self._update_stats(processing_time, len(document.chunks))
            
            logging.info(f"Document processed successfully: {filename} ({len(document.chunks)} chunks, {processing_time:.2f}s)")
            
            return document.id
            
        except Exception as e:
            logging.error(f"Error processing document {filename}: {e}")
            raise
    
    def upload_document(self, content: str, filename: str) -> str:
        """Upload and process a document from text content (backward compatibility)"""
        start_time = time.time()
        
        try:
            # Use enhanced document processing if possible
            if self.is_supported_format(filename):
                # Create temporary file for processing
                with tempfile.NamedTemporaryFile(mode='w', suffix=Path(filename).suffix, delete=False) as tmp_file:
                    tmp_file.write(content)
                    tmp_path = tmp_file.name
                
                try:
                    result = self.upload_document_file(tmp_path, filename)
                finally:
                    os.unlink(tmp_path)
                
                return result
            else:
                # Fallback to original text processing
                document = self.text_processor.create_document(content, filename)
                document.chunks = self.text_processor.chunk_text(content)
                
                # Generate embeddings
                chunk_texts = [chunk.content for chunk in document.chunks]
                embeddings = self.embedding_service.generate_embeddings(chunk_texts)
                
                # Store in vector database
                self.vector_store.store_document(document, embeddings)
                
                # Update stats
                processing_time = time.time() - start_time
                self._update_stats(processing_time, len(document.chunks))
                
                return document.id
            
        except Exception as e:
            logging.error(f"Error processing text content: {e}")
            raise
    
    def upload_document_bytes(self, file_content: bytes, filename: str) -> str:
        """Upload and process a document from bytes (enhanced)"""
        start_time = time.time()
        
        if not self.is_supported_format(filename):
            raise ValueError(f"Unsupported file format: {Path(filename).suffix}")
        
        try:
            # Process document
            document = self.document_processor.process_bytes(file_content, filename)
            
            # Generate embeddings
            chunk_texts = [chunk.content for chunk in document.chunks]
            embeddings = self.embedding_service.generate_embeddings(chunk_texts)
            
            # Store in vector database
            success = self.vector_store.store_document(document, embeddings)
            
            if not success:
                raise RuntimeError("Failed to store document in vector database")
            
            # Update stats
            processing_time = time.time() - start_time
            self._update_stats(processing_time, len(document.chunks))
            
            logging.info(f"Document processed successfully: {filename} ({len(document.chunks)} chunks, {processing_time:.2f}s)")
            
            return document.id
            
        except Exception as e:
            logging.error(f"Error processing document bytes {filename}: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5, 
              filters: Optional[Dict[str, Any]] = None,
              document_type: Optional[str] = None,
              author: Optional[str] = None) -> List[Dict[str, Any]]:
        """Enhanced search with filtering capabilities"""
        start_time = time.time()
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        query = query.strip()
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Apply filters if supported
            if hasattr(self.vector_store, 'search_with_filters') and filters:
                results = self.vector_store.search_with_filters(query_embedding, filters, top_k)
            elif hasattr(self.vector_store, 'search_by_document_type') and document_type:
                results = self.vector_store.search_by_document_type(query_embedding, document_type, top_k)
            elif hasattr(self.vector_store, 'search_by_author') and author:
                results = self.vector_store.search_by_author(query_embedding, author, top_k)
            else:
                # Standard search
                results = self.vector_store.search_similar(query_embedding, top_k)
            
            # Update search stats
            search_time = time.time() - start_time
            self.stats['searches_performed'] += 1
            
            logging.debug(f"Search completed: {len(results)} results in {search_time:.3f}s")
            
            return results
            
        except Exception as e:
            logging.error(f"Error performing search: {e}")
            raise
    
    def search_by_type(self, query: str, document_type: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search within specific document type"""
        return self.search(query, top_k, document_type=document_type)
    
    def search_by_author(self, query: str, author: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search within documents by specific author"""
        return self.search(query, top_k, author=author)
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all uploaded documents with enhanced metadata"""
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
    
    def answer_question(self, question: str, max_context_chunks: int = 3,
                       filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Answer a question based on stored documents with enhanced context"""
        search_results = self.search(question, top_k=max_context_chunks, filters=filters)
        
        if not search_results:
            return {
                'answer': "I don't have information to answer that question.",
                'confidence': 0.0,
                'sources': []
            }
        
        # Enhanced answer generation
        top_result = search_results[0]
        
        # Build context from multiple chunks
        context_parts = []
        for result in search_results:
            context_parts.append(f"[From {result['filename']}]: {result['content']}")
        
        context = "\n\n".join(context_parts)
        
        # Simple answer generation (can be enhanced with language models)
        answer = f"Based on the available information:\n\n{context}"
        
        return {
            'answer': answer,
            'confidence': top_result['similarity'],
            'context_chunks': len(search_results),
            'sources': [{
                'content': result['content'],
                'filename': result['filename'],
                'similarity': result['similarity'],
                'document_id': result.get('document_id'),
                'document_type': result.get('document_type')
            } for result in search_results]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge base"""
        documents = self.get_all_documents()
        
        # Basic stats
        total_chunks = 0
        document_types = {}
        authors = set()
        
        for doc in documents:
            chunks = self.vector_store.get_document_chunks(doc['id'])
            total_chunks += len(chunks)
            
            # Count document types
            doc_type = doc.get('document_type', 'unknown')
            document_types[doc_type] = document_types.get(doc_type, 0) + 1
            
            # Collect authors
            author = doc.get('author')
            if author:
                authors.add(author)
        
        # Collection stats
        collection_stats = {}
        if hasattr(self.vector_store, 'get_collection_stats'):
            collection_stats = self.vector_store.get_collection_stats()
        
        # Embedding service stats
        embedding_stats = {}
        if hasattr(self.embedding_service, 'get_model_info'):
            embedding_stats = self.embedding_service.get_model_info()
        
        return {
            'total_documents': len(documents),
            'total_chunks': total_chunks,
            'average_chunks_per_document': total_chunks / len(documents) if documents else 0,
            'document_types': document_types,
            'unique_authors': len(authors),
            'processing_stats': self.stats.copy(),
            'collection_stats': collection_stats,
            'embedding_stats': embedding_stats,
            'supported_formats': self.get_supported_formats()
        }
    
    def get_unique_document_types(self) -> List[str]:
        """Get list of unique document types"""
        if hasattr(self.vector_store, 'get_unique_document_types'):
            return self.vector_store.get_unique_document_types()
        else:
            # Fallback implementation
            documents = self.get_all_documents()
            types = set()
            for doc in documents:
                doc_type = doc.get('document_type', 'unknown')
                types.add(doc_type)
            return sorted(list(types))
    
    def get_unique_authors(self) -> List[str]:
        """Get list of unique authors"""
        if hasattr(self.vector_store, 'get_unique_authors'):
            return self.vector_store.get_unique_authors()
        else:
            # Fallback implementation
            documents = self.get_all_documents()
            authors = set()
            for doc in documents:
                author = doc.get('author')
                if author:
                    authors.add(author)
            return sorted(list(authors))
    
    def benchmark_performance(self, sample_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Benchmark the RAG service performance"""
        if sample_texts is None:
            sample_texts = [
                "This is a test document about machine learning and artificial intelligence.",
                "Natural language processing is a field that combines computational linguistics with machine learning.",
                "Deep learning models use neural networks to process complex patterns in large datasets."
            ]
        
        start_time = time.time()
        
        # Test document upload
        upload_times = []
        doc_ids = []
        
        for i, text in enumerate(sample_texts):
            upload_start = time.time()
            doc_id = self.upload_document(text, f"benchmark_doc_{i}.txt")
            upload_time = time.time() - upload_start
            
            upload_times.append(upload_time)
            doc_ids.append(doc_id)
        
        # Test search
        search_times = []
        search_results_counts = []
        
        for query in ["machine learning", "natural language", "neural networks"]:
            search_start = time.time()
            results = self.search(query, top_k=5)
            search_time = time.time() - search_start
            
            search_times.append(search_time)
            search_results_counts.append(len(results))
        
        # Test embedding service if available
        embedding_benchmark = {}
        if hasattr(self.embedding_service, 'benchmark'):
            embedding_benchmark = self.embedding_service.benchmark(sample_texts)
        
        total_time = time.time() - start_time
        
        # Cleanup benchmark documents
        for doc_id in doc_ids:
            self.delete_document(doc_id)
        
        return {
            'total_benchmark_time': total_time,
            'upload_performance': {
                'average_upload_time': sum(upload_times) / len(upload_times),
                'min_upload_time': min(upload_times),
                'max_upload_time': max(upload_times),
                'documents_uploaded': len(upload_times)
            },
            'search_performance': {
                'average_search_time': sum(search_times) / len(search_times),
                'min_search_time': min(search_times),
                'max_search_time': max(search_times),
                'average_results_count': sum(search_results_counts) / len(search_results_counts)
            },
            'embedding_benchmark': embedding_benchmark
        }
    
    def _update_stats(self, processing_time: float, chunks_created: int):
        """Update processing statistics"""
        self.stats['documents_processed'] += 1
        self.stats['chunks_created'] += chunks_created
        self.stats['total_processing_time'] += processing_time
        self.stats['average_processing_time'] = (
            self.stats['total_processing_time'] / self.stats['documents_processed']
        )
    
    def backup_data(self, backup_path: str) -> bool:
        """Backup vector database data"""
        if hasattr(self.vector_store, 'backup_to_sqlite'):
            return self.vector_store.backup_to_sqlite(backup_path)
        else:
            logging.warning("Backup not supported by current vector store")
            return False
    
    def clear_cache(self):
        """Clear embedding cache"""
        if hasattr(self.embedding_service, 'clear_cache'):
            self.embedding_service.clear_cache()
    
    def close(self):
        """Clean up resources"""
        if hasattr(self.vector_store, 'close'):
            self.vector_store.close()
        
        logging.info("Enhanced RAG service closed")