"""
Enhanced vector store using ChromaDB for better scalability and performance.
Maintains backward compatibility with the original SQLite-based store.
"""
import sqlite3
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
import os
import uuid
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available. Falling back to SQLite vector store.")

from .vector_store import VectorStore as SQLiteVectorStore
try:
    from ..core.models import Document
except ImportError:
    from core.models import Document


class ChromaVectorStore:
    """Enhanced vector store using ChromaDB for scalability"""
    
    def __init__(self, 
                 persist_directory: str = "chroma_db",
                 collection_name: str = "rag_documents",
                 fallback_db_path: Optional[str] = None):
        """
        Initialize ChromaDB vector store
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            fallback_db_path: SQLite database path for fallback/metadata
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        
        # Ensure persist directory exists
        self.persist_directory.mkdir(exist_ok=True)
        
        # Initialize ChromaDB
        self.chroma_client = None
        self.collection = None
        
        if CHROMADB_AVAILABLE:
            try:
                self._initialize_chromadb()
                logging.info(f"ChromaDB initialized successfully: {persist_directory}")
            except Exception as e:
                logging.error(f"Failed to initialize ChromaDB: {e}")
                self.chroma_client = None
        
        # Initialize SQLite fallback for metadata and compatibility
        fallback_path = fallback_db_path or str(self.persist_directory / "metadata.db")
        self.metadata_store = SQLiteVectorStore(fallback_path)
        
        # Track which backend to use
        self.use_chroma = self.chroma_client is not None
        
        if not self.use_chroma:
            logging.warning("Using SQLite fallback mode")
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection"""
        # Create persistent client
        settings = Settings(
            persist_directory=str(self.persist_directory),
            anonymized_telemetry=False
        )
        
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=settings
        )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            logging.info(f"Connected to existing collection: {self.collection_name}")
        except:
            # Collection doesn't exist, create it
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logging.info(f"Created new collection: {self.collection_name}")
    
    def initialize_database(self):
        """Initialize database (compatibility method)"""
        if not self.use_chroma:
            self.metadata_store.initialize_database()
        # ChromaDB initializes automatically
    
    def store_document(self, document: Document, embeddings: List[np.ndarray]) -> bool:
        """Store document with its chunks and embeddings"""
        if len(embeddings) != len(document.chunks):
            raise ValueError(f"Number of embeddings ({len(embeddings)}) must match number of chunks ({len(document.chunks)})")
        
        if self.use_chroma:
            return self._store_document_chroma(document, embeddings)
        else:
            return self.metadata_store.store_document(document, embeddings)
    
    def _store_document_chroma(self, document: Document, embeddings: List[np.ndarray]) -> bool:
        """Store document in ChromaDB"""
        try:
            # Prepare data for ChromaDB
            ids = []
            documents_text = []
            metadatas = []
            embeddings_list = []
            
            for i, (chunk, embedding) in enumerate(zip(document.chunks, embeddings)):
                chunk_id = chunk.chunk_id
                ids.append(chunk_id)
                documents_text.append(chunk.content)
                
                # Prepare metadata
                metadata = {
                    'document_id': document.id,
                    'filename': document.filename,
                    'chunk_index': i,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'document_type': getattr(document, 'document_type', 'unknown')
                }
                
                # Add enhanced metadata if available
                if hasattr(document, 'metadata'):
                    doc_meta = document.metadata
                    if doc_meta.title:
                        metadata['title'] = doc_meta.title
                    if doc_meta.author:
                        metadata['author'] = doc_meta.author
                    if doc_meta.creation_date:
                        metadata['creation_date'] = doc_meta.creation_date
                
                metadatas.append(metadata)
                embeddings_list.append(embedding.tolist())
            
            # Remove existing chunks for this document
            try:
                existing_results = self.collection.get(
                    where={"document_id": document.id}
                )
                if existing_results['ids']:
                    self.collection.delete(ids=existing_results['ids'])
            except Exception as e:
                logging.warning(f"Could not delete existing chunks: {e}")
            
            # Add new chunks
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            
            # Also store in metadata SQLite for backward compatibility
            try:
                self.metadata_store.initialize_database()
                self.metadata_store.store_document(document, embeddings)
            except Exception as e:
                logging.warning(f"Failed to store in metadata SQLite: {e}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error storing document in ChromaDB: {e}")
            return False
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5, 
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar chunks"""
        if self.use_chroma:
            return self._search_similar_chroma(query_embedding, top_k, filters)
        else:
            return self.metadata_store.search_similar(query_embedding, top_k)
    
    def _search_similar_chroma(self, query_embedding: np.ndarray, top_k: int = 5,
                              filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search using ChromaDB"""
        try:
            # Prepare query
            query_embeddings = [query_embedding.tolist()]
            
            # Execute query
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k,
                where=filters,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            
            if results['ids'] and results['ids'][0]:  # Check if we have results
                for i in range(len(results['ids'][0])):
                    result = {
                        'chunk_id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'similarity': 1.0 - results['distances'][0][i],  # Convert distance to similarity
                        'document_id': results['metadatas'][0][i]['document_id'],
                        'filename': results['metadatas'][0][i]['filename']
                    }
                    
                    # Add additional metadata if available
                    metadata = results['metadatas'][0][i]
                    if 'title' in metadata:
                        result['title'] = metadata['title']
                    if 'author' in metadata:
                        result['author'] = metadata['author']
                    if 'document_type' in metadata:
                        result['document_type'] = metadata['document_type']
                    
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logging.error(f"Error searching in ChromaDB: {e}")
            return []
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        if self.use_chroma:
            return self._get_document_chunks_chroma(document_id)
        else:
            return self.metadata_store.get_document_chunks(document_id)
    
    def _get_document_chunks_chroma(self, document_id: str) -> List[Dict[str, Any]]:
        """Get document chunks from ChromaDB"""
        try:
            results = self.collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas"]
            )
            
            chunks = []
            if results['ids']:
                for i, chunk_id in enumerate(results['ids']):
                    chunk_data = {
                        'chunk_id': chunk_id,
                        'content': results['documents'][i],
                        'start_index': results['metadatas'][i].get('start_index', 0),
                        'end_index': results['metadatas'][i].get('end_index', 0)
                    }
                    chunks.append(chunk_data)
                
                # Sort by chunk_index to maintain order
                chunks.sort(key=lambda x: results['metadatas'][results['ids'].index(x['chunk_id'])].get('chunk_index', 0))
            
            return chunks
            
        except Exception as e:
            logging.error(f"Error getting document chunks from ChromaDB: {e}")
            return []
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all stored documents"""
        if self.use_chroma:
            return self._get_all_documents_chroma()
        else:
            return self.metadata_store.get_all_documents()
    
    def _get_all_documents_chroma(self) -> List[Dict[str, Any]]:
        """Get all documents from ChromaDB metadata"""
        try:
            # Get all unique documents by querying metadata
            all_results = self.collection.get(include=["metadatas"])
            
            # Group by document_id
            documents = {}
            for metadata in all_results['metadatas']:
                doc_id = metadata['document_id']
                if doc_id not in documents:
                    documents[doc_id] = {
                        'id': doc_id,
                        'filename': metadata['filename'],
                        'document_type': metadata.get('document_type', 'unknown'),
                        'title': metadata.get('title'),
                        'author': metadata.get('author'),
                        'created_at': metadata.get('creation_date')
                    }
            
            return list(documents.values())
            
        except Exception as e:
            logging.error(f"Error getting all documents from ChromaDB: {e}")
            # Fallback to metadata store
            try:
                return self.metadata_store.get_all_documents()
            except:
                return []
    
    def delete_document(self, document_id: str) -> bool:
        """Delete document and all its chunks"""
        if self.use_chroma:
            return self._delete_document_chroma(document_id)
        else:
            return self.metadata_store.delete_document(document_id)
    
    def _delete_document_chroma(self, document_id: str) -> bool:
        """Delete document from ChromaDB"""
        try:
            # Get all chunk IDs for this document
            results = self.collection.get(where={"document_id": document_id})
            
            if results['ids']:
                # Delete chunks from ChromaDB
                self.collection.delete(ids=results['ids'])
                
                # Also delete from metadata store
                try:
                    self.metadata_store.delete_document(document_id)
                except Exception as e:
                    logging.warning(f"Failed to delete from metadata store: {e}")
                
                return True
            else:
                return False  # Document not found
                
        except Exception as e:
            logging.error(f"Error deleting document from ChromaDB: {e}")
            return False
    
    def update_document(self, document: Document, embeddings: List[np.ndarray]) -> bool:
        """Update existing document"""
        return self.store_document(document, embeddings)  # ChromaDB handles updates via upsert
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        if self.use_chroma:
            try:
                count = self.collection.count()
                return {
                    'total_chunks': count,
                    'backend': 'chromadb',
                    'collection_name': self.collection_name
                }
            except Exception:
                return {'backend': 'chromadb', 'error': 'Could not get stats'}
        else:
            # Use SQLite stats
            try:
                docs = self.metadata_store.get_all_documents()
                return {
                    'total_documents': len(docs),
                    'backend': 'sqlite_fallback'
                }
            except:
                return {'backend': 'sqlite_fallback', 'error': 'Could not get stats'}
    
    def search_with_filters(self, query_embedding: np.ndarray, 
                           filters: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search with metadata filters (enhanced functionality)"""
        if self.use_chroma:
            return self._search_similar_chroma(query_embedding, top_k, filters)
        else:
            # Basic search without filters for SQLite fallback
            return self.metadata_store.search_similar(query_embedding, top_k)
    
    def search_by_document_type(self, query_embedding: np.ndarray, 
                               document_type: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search within specific document type"""
        filters = {"document_type": document_type}
        return self.search_with_filters(query_embedding, filters, top_k)
    
    def search_by_author(self, query_embedding: np.ndarray, 
                        author: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search within documents by specific author"""
        filters = {"author": author}
        return self.search_with_filters(query_embedding, filters, top_k)
    
    def get_unique_authors(self) -> List[str]:
        """Get list of unique authors in the collection"""
        if self.use_chroma:
            try:
                all_results = self.collection.get(include=["metadatas"])
                authors = set()
                for metadata in all_results['metadatas']:
                    author = metadata.get('author')
                    if author:
                        authors.add(author)
                return sorted(list(authors))
            except Exception:
                return []
        else:
            return []  # Not supported in SQLite fallback
    
    def get_unique_document_types(self) -> List[str]:
        """Get list of unique document types in the collection"""
        if self.use_chroma:
            try:
                all_results = self.collection.get(include=["metadatas"])
                types = set()
                for metadata in all_results['metadatas']:
                    doc_type = metadata.get('document_type', 'unknown')
                    types.add(doc_type)
                return sorted(list(types))
            except Exception:
                return []
        else:
            return []  # Not supported in SQLite fallback
    
    def close(self):
        """Clean up resources"""
        if self.metadata_store:
            self.metadata_store.close()
        
        # ChromaDB doesn't need explicit closing
        self.chroma_client = None
        self.collection = None
    
    def backup_to_sqlite(self, backup_path: str) -> bool:
        """Backup ChromaDB data to SQLite for portability"""
        if not self.use_chroma:
            return False
        
        try:
            # Create backup SQLite store
            backup_store = SQLiteVectorStore(backup_path)
            backup_store.initialize_database()
            
            # Get all documents and their chunks
            documents = self.get_all_documents()
            
            for doc_info in documents:
                doc_id = doc_info['id']
                
                # Get chunks with embeddings
                results = self.collection.get(
                    where={"document_id": doc_id},
                    include=["documents", "metadatas", "embeddings"]
                )
                
                if results['ids']:
                    # Reconstruct document
                    content = "\n".join(results['documents'])
                    doc = Document(content, doc_info['filename'])
                    doc.id = doc_id
                    
                    # Reconstruct chunks
                    try:
                        from ..core.models import Chunk
                    except ImportError:
                        from core.models import Chunk
                    chunks = []
                    embeddings = []
                    
                    for i, chunk_id in enumerate(results['ids']):
                        chunk = Chunk(
                            content=results['documents'][i],
                            start_index=results['metadatas'][i].get('start_index', 0),
                            end_index=results['metadatas'][i].get('end_index', 0)
                        )
                        chunk.chunk_id = chunk_id
                        chunks.append(chunk)
                        embeddings.append(np.array(results['embeddings'][i]))
                    
                    doc.chunks = chunks
                    
                    # Store in backup
                    backup_store.store_document(doc, embeddings)
            
            backup_store.close()
            return True
            
        except Exception as e:
            logging.error(f"Error backing up to SQLite: {e}")
            return False