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
    """Enhanced vector store using ChromaDB with FASE 2 cross-document correlation capabilities"""
    
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
        
        # Initialize FASE 2 components
        self.relationship_manager = None
        self.quality_monitor = None
        
        if self.use_chroma:
            self._initialize_fase2_components()
        
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
                    
                    # Add FASE 2 enhanced metadata
                    if hasattr(doc_meta, 'quality_score'):
                        metadata['quality_score'] = doc_meta.quality_score
                    if hasattr(doc_meta, 'extraction_confidence'):
                        metadata['extraction_confidence'] = doc_meta.extraction_confidence.overall
                    if hasattr(doc_meta, 'processing_stats'):
                        metadata['processing_time'] = doc_meta.processing_stats.total_processing_time
                
                # Add enhanced chunk metadata for hierarchical search
                if hasattr(chunk, 'chunk_type'):
                    metadata['chunk_type'] = chunk.chunk_type.value if hasattr(chunk.chunk_type, 'value') else str(chunk.chunk_type)
                if hasattr(chunk, 'hierarchy_level'):
                    metadata['hierarchy_level'] = chunk.hierarchy_level
                if hasattr(chunk, 'parent_section'):
                    metadata['parent_section'] = chunk.parent_section
                if hasattr(chunk, 'semantic_topics'):
                    metadata['semantic_topics'] = json.dumps(chunk.semantic_topics)
                if hasattr(chunk, 'confidence'):
                    metadata['chunk_confidence'] = chunk.confidence
                
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
            
            # Update quality metrics if FASE 2 components are available
            if self.quality_monitor and hasattr(document, 'metadata'):
                quality_metrics = {
                    'quality_score': getattr(document.metadata, 'quality_score', 0.0),
                    'extraction_confidence': getattr(document.metadata.extraction_confidence, 'overall', 0.0) if hasattr(document.metadata, 'extraction_confidence') else 0.0,
                    'processing_time': getattr(document.metadata.processing_stats, 'total_processing_time', 0.0) if hasattr(document.metadata, 'processing_stats') else 0.0,
                    'chunk_count': len(document.chunks),
                    'validation_status': 'stored',
                    'metadata': {
                        'document_type': getattr(document, 'document_type', 'unknown'),
                        'filename': document.filename,
                        'has_hierarchy': hasattr(document.metadata, 'hierarchy_tree') and document.metadata.hierarchy_tree is not None,
                        'has_tables': len(getattr(document.metadata, 'table_schemas', [])) > 0,
                        'has_figures': len(getattr(document.metadata, 'figures', [])) > 0
                    }
                }
                
                try:
                    self.quality_monitor.update_document_quality(document.id, quality_metrics)
                except Exception as e:
                    logging.warning(f"Failed to update quality metrics: {e}")
            
            # Trigger correlation analysis for FASE 2
            if self.relationship_manager:
                try:
                    correlation_result = FASE2Helper.trigger_correlation_update(self, document.id)
                    if correlation_result.get('status') == 'completed':
                        logging.info(f"Correlation analysis completed: {correlation_result}")
                except Exception as e:
                    logging.warning(f"Failed to trigger correlation analysis: {e}")
            
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
    
    def _initialize_fase2_components(self):
        """Initialize FASE 2 cross-document correlation components"""
        try:
            self.relationship_manager = DocumentRelationshipManager(
                self.chroma_client, 
                f"{self.collection_name}_relationships"
            )
            self.quality_monitor = QualityMonitor(self.persist_directory / "quality.db")
            logging.info("FASE 2 components initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize FASE 2 components: {e}")
            self.relationship_manager = None
            self.quality_monitor = None
    
    def close(self):
        """Clean up resources"""
        if self.metadata_store:
            self.metadata_store.close()
            
        if self.relationship_manager:
            self.relationship_manager.close()
            
        if self.quality_monitor:
            self.quality_monitor.close()
        
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
    
    # FASE 2 Cross-Document Correlation Methods
    
    def search_with_cross_document_correlation(self, query_embedding: np.ndarray, 
                                             top_k: int = 5, 
                                             correlation_threshold: float = 0.7,
                                             include_related: bool = True) -> List[Dict[str, Any]]:
        """Enhanced search with cross-document correlation capabilities"""
        if not self.use_chroma or not self.relationship_manager:
            return self.search_similar(query_embedding, top_k)
        
        try:
            # Get initial search results
            initial_results = self.search_similar(query_embedding, top_k)
            
            if not include_related:
                return initial_results
            
            # Get cross-document correlations
            enhanced_results = []
            document_ids = [result['document_id'] for result in initial_results]
            
            for result in initial_results:
                enhanced_result = result.copy()
                
                # Add related documents
                related_docs = self.relationship_manager.get_related_documents(
                    result['document_id'], correlation_threshold
                )
                enhanced_result['related_documents'] = related_docs
                
                # Add confidence scoring
                confidence = self._calculate_result_confidence(result, query_embedding)
                enhanced_result['confidence_score'] = confidence
                
                # Add document hierarchy context if available
                hierarchy_context = self._get_hierarchy_context(result['document_id'], result['chunk_id'])
                if hierarchy_context:
                    enhanced_result['hierarchy_context'] = hierarchy_context
                
                enhanced_results.append(enhanced_result)
            
            # Sort by combined relevance and confidence
            enhanced_results.sort(
                key=lambda x: x['similarity'] * 0.7 + x.get('confidence_score', 0.0) * 0.3,
                reverse=True
            )
            
            return enhanced_results
            
        except Exception as e:
            logging.error(f"Error in cross-document correlation search: {e}")
            return initial_results
    
    def search_within_document_hierarchy(self, query_embedding: np.ndarray,
                                       document_id: Optional[str] = None,
                                       section_filter: Optional[str] = None,
                                       hierarchy_level: Optional[int] = None,
                                       top_k: int = 5) -> List[Dict[str, Any]]:
        """Search within specific document hierarchy levels"""
        if not self.use_chroma:
            return self.search_similar(query_embedding, top_k)
        
        try:
            filters = {}
            
            if document_id:
                filters['document_id'] = document_id
            
            if section_filter:
                filters['parent_section'] = section_filter
                
            if hierarchy_level is not None:
                filters['hierarchy_level'] = hierarchy_level
            
            results = self.search_with_filters(query_embedding, filters, top_k)
            
            # Enhance results with section context
            for result in results:
                if 'parent_section' in result:
                    # Add section navigation links
                    result['section_navigation'] = self._get_section_navigation(
                        result['document_id'], result['parent_section']
                    )
            
            return results
            
        except Exception as e:
            logging.error(f"Error in hierarchical search: {e}")
            return self.search_similar(query_embedding, top_k)
    
    def search_by_document_correlation_cluster(self, query_embedding: np.ndarray,
                                             cluster_id: Optional[str] = None,
                                             correlation_type: str = "semantic",
                                             top_k: int = 5) -> List[Dict[str, Any]]:
        """Search within document correlation clusters"""
        if not self.use_chroma or not self.relationship_manager:
            return self.search_similar(query_embedding, top_k)
        
        try:
            if cluster_id:
                # Search within specific cluster
                cluster_docs = self.relationship_manager.get_cluster_documents(cluster_id)
                filters = {'document_id': {'$in': cluster_docs}}
            else:
                # Find best matching cluster first
                initial_results = self.search_similar(query_embedding, min(top_k, 3))
                if not initial_results:
                    return []
                
                # Get cluster for the best match
                best_doc_id = initial_results[0]['document_id']
                cluster_docs = self.relationship_manager.find_document_cluster(
                    best_doc_id, correlation_type
                )
                filters = {'document_id': {'$in': cluster_docs}}
            
            results = self.search_with_filters(query_embedding, filters, top_k)
            
            # Add cluster context
            for result in results:
                result['cluster_context'] = {
                    'cluster_id': cluster_id,
                    'correlation_type': correlation_type,
                    'cluster_size': len(cluster_docs) if 'cluster_docs' in locals() else 0
                }
            
            return results
            
        except Exception as e:
            logging.error(f"Error in cluster-based search: {e}")
            return self.search_similar(query_embedding, top_k)
    
    def get_document_citation_network(self, document_id: str) -> Dict[str, Any]:
        """Get citation network for a document"""
        if not self.relationship_manager:
            return {'citations': [], 'references': [], 'network_depth': 0}
        
        try:
            return self.relationship_manager.get_citation_network(document_id)
        except Exception as e:
            logging.error(f"Error getting citation network: {e}")
            return {'citations': [], 'references': [], 'network_depth': 0}
    
    def analyze_cross_document_dependencies(self, document_ids: List[str]) -> Dict[str, Any]:
        """Analyze dependencies between documents"""
        if not self.relationship_manager:
            return {'dependencies': {}, 'dependency_graph': {}, 'circular_dependencies': []}
        
        try:
            return self.relationship_manager.analyze_dependencies(document_ids)
        except Exception as e:
            logging.error(f"Error analyzing dependencies: {e}")
            return {'dependencies': {}, 'dependency_graph': {}, 'circular_dependencies': []}
    
    def get_quality_metrics(self, document_id: Optional[str] = None) -> Dict[str, Any]:
        """Get quality metrics for documents or entire collection"""
        if not self.quality_monitor:
            return {'quality_score': 0.0, 'confidence_metrics': {}, 'validation_status': 'unknown'}
        
        try:
            if document_id:
                return self.quality_monitor.get_document_quality(document_id)
            else:
                return self.quality_monitor.get_collection_quality()
        except Exception as e:
            logging.error(f"Error getting quality metrics: {e}")
            return {'quality_score': 0.0, 'confidence_metrics': {}, 'validation_status': 'unknown'}
    
    def validate_correlations(self, correlation_threshold: float = 0.8) -> Dict[str, Any]:
        """Validate cross-document correlations and relationships"""
        if not self.relationship_manager or not self.quality_monitor:
            return {'validation_status': 'unavailable', 'issues': ['FASE 2 components not initialized']}
        
        try:
            return self.relationship_manager.validate_relationships(correlation_threshold)
        except Exception as e:
            logging.error(f"Error validating correlations: {e}")
            return {'validation_status': 'error', 'issues': [str(e)]}
    
    def rebuild_correlation_index(self) -> bool:
        """Rebuild cross-document correlation index"""
        if not self.relationship_manager:
            logging.warning("Relationship manager not available")
            return False
        
        try:
            return self.relationship_manager.rebuild_index()
        except Exception as e:
            logging.error(f"Error rebuilding correlation index: {e}")
            return False
    
    def _calculate_result_confidence(self, result: Dict[str, Any], 
                                   query_embedding: np.ndarray) -> float:
        """Calculate confidence score for search result"""
        try:
            base_similarity = result.get('similarity', 0.0)
            
            # Adjust confidence based on document quality if available
            quality_score = 1.0
            if self.quality_monitor:
                doc_quality = self.quality_monitor.get_document_quality(result['document_id'])
                quality_score = doc_quality.get('quality_score', 1.0)
            
            # Adjust confidence based on chunk type and hierarchy
            chunk_confidence = 1.0
            if 'chunk_type' in result:
                chunk_type_weights = {
                    'header': 0.9,
                    'paragraph': 1.0,
                    'table': 0.8,
                    'figure': 0.7,
                    'list': 0.85,
                    'code': 0.9
                }
                chunk_confidence = chunk_type_weights.get(result['chunk_type'], 1.0)
            
            # Combined confidence score
            confidence = base_similarity * 0.6 + quality_score * 0.3 + chunk_confidence * 0.1
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logging.warning(f"Error calculating result confidence: {e}")
            return result.get('similarity', 0.0)
    
    def _get_hierarchy_context(self, document_id: str, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get hierarchy context for a chunk"""
        try:
            # Get chunk metadata to determine hierarchy position
            chunk_results = self.collection.get(
                ids=[chunk_id],
                include=["metadatas"]
            )
            
            if not chunk_results['metadatas']:
                return None
            
            metadata = chunk_results['metadatas'][0]
            
            context = {
                'parent_section': metadata.get('parent_section', 'root'),
                'hierarchy_level': metadata.get('hierarchy_level', 0),
                'chunk_type': metadata.get('chunk_type', 'paragraph')
            }
            
            # Get sibling chunks at the same level
            sibling_results = self.collection.get(
                where={
                    'document_id': document_id,
                    'parent_section': context['parent_section']
                },
                include=["metadatas"]
            )
            
            context['siblings_count'] = len(sibling_results['ids']) if sibling_results['ids'] else 0
            
            return context
            
        except Exception as e:
            logging.warning(f"Error getting hierarchy context: {e}")
            return None
    
    def _get_section_navigation(self, document_id: str, section: str) -> Dict[str, Any]:
        """Get section navigation information"""
        try:
            # Find all sections in the document
            doc_results = self.collection.get(
                where={'document_id': document_id},
                include=["metadatas"]
            )
            
            if not doc_results['metadatas']:
                return {'parent': None, 'children': [], 'siblings': []}
            
            # Extract section hierarchy
            sections = {}
            for metadata in doc_results['metadatas']:
                parent_section = metadata.get('parent_section', 'root')
                level = metadata.get('hierarchy_level', 0)
                
                if parent_section not in sections:
                    sections[parent_section] = {'level': level, 'children': []}
                
                section_name = metadata.get('section_name', f"section_{len(sections)}")
                if section_name != parent_section:
                    sections[parent_section]['children'].append(section_name)
            
            # Find navigation for current section
            current_level = sections.get(section, {}).get('level', 0)
            parent = None
            siblings = []
            children = sections.get(section, {}).get('children', [])
            
            # Find parent and siblings
            for sec, info in sections.items():
                if section in info['children']:
                    parent = sec
                    siblings = [child for child in info['children'] if child != section]
                    break
            
            return {
                'parent': parent,
                'children': children,
                'siblings': siblings,
                'current_level': current_level
            }
            
        except Exception as e:
            logging.warning(f"Error getting section navigation: {e}")
            return {'parent': None, 'children': [], 'siblings': []}


class DocumentRelationshipManager:
    """Manages cross-document relationships and correlations"""
    
    def __init__(self, chroma_client, collection_name: str):
        self.chroma_client = chroma_client
        self.collection_name = collection_name
        self.relationships_collection = None
        self.similarity_threshold = 0.7
        self.max_correlation_depth = 3
        
        try:
            self._initialize_relationships_collection()
        except Exception as e:
            logging.error(f"Failed to initialize relationships collection: {e}")
    
    def _initialize_relationships_collection(self):
        """Initialize the relationships collection"""
        try:
            self.relationships_collection = self.chroma_client.get_collection(
                name=self.collection_name
            )
            logging.info(f"Connected to relationships collection: {self.collection_name}")
        except:
            self.relationships_collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine", "purpose": "document_relationships"}
            )
            logging.info(f"Created relationships collection: {self.collection_name}")
    
    def build_semantic_clusters(self, documents_collection) -> Dict[str, Any]:
        """Build semantic clusters of related documents"""
        if not SKLEARN_AVAILABLE:
            logging.warning("sklearn not available, using simple similarity clustering")
            return self._build_simple_clusters(documents_collection)
        
        try:
            # Get all document embeddings
            all_results = documents_collection.get(include=["embeddings", "metadatas"])
            
            if not all_results['embeddings']:
                return {'clusters': {}, 'cluster_assignments': {}}
            
            embeddings = np.array(all_results['embeddings'])
            document_ids = [meta['document_id'] for meta in all_results['metadatas']]
            
            # Use DBSCAN for clustering
            clustering = DBSCAN(
                eps=1 - self.similarity_threshold,  # Convert similarity to distance
                min_samples=2,
                metric='cosine'
            )
            
            cluster_labels = clustering.fit_predict(embeddings)
            
            # Group documents by cluster
            clusters = {}
            cluster_assignments = {}
            
            for doc_id, cluster_id in zip(document_ids, cluster_labels):
                cluster_id = int(cluster_id)  # Convert numpy int to Python int
                
                if cluster_id == -1:  # Noise points
                    cluster_id = f"singleton_{doc_id}"
                
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                
                clusters[cluster_id].append(doc_id)
                cluster_assignments[doc_id] = cluster_id
            
            # Store cluster information
            self._store_cluster_relationships(clusters)
            
            logging.info(f"Built {len(clusters)} semantic clusters")
            
            return {
                'clusters': clusters,
                'cluster_assignments': cluster_assignments,
                'cluster_count': len(clusters),
                'singleton_count': sum(1 for k in clusters.keys() if str(k).startswith('singleton'))
            }
            
        except Exception as e:
            logging.error(f"Error building semantic clusters: {e}")
            return self._build_simple_clusters(documents_collection)
    
    def _build_simple_clusters(self, documents_collection) -> Dict[str, Any]:
        """Simple clustering fallback without sklearn"""
        try:
            # Get document metadata and group by type/author
            all_results = documents_collection.get(include=["metadatas"])
            
            clusters = {}
            cluster_assignments = {}
            
            # Cluster by document type
            for i, metadata in enumerate(all_results['metadatas']):
                doc_id = metadata['document_id']
                doc_type = metadata.get('document_type', 'unknown')
                author = metadata.get('author', 'unknown')
                
                cluster_key = f"{doc_type}_{author}"
                
                if cluster_key not in clusters:
                    clusters[cluster_key] = []
                
                clusters[cluster_key].append(doc_id)
                cluster_assignments[doc_id] = cluster_key
            
            return {
                'clusters': clusters,
                'cluster_assignments': cluster_assignments,
                'cluster_count': len(clusters),
                'method': 'simple_metadata'
            }
            
        except Exception as e:
            logging.error(f"Error in simple clustering: {e}")
            return {'clusters': {}, 'cluster_assignments': {}}
    
    def _store_cluster_relationships(self, clusters: Dict[str, List[str]]):
        """Store cluster relationship information"""
        try:
            # Store relationships between documents in the same cluster
            for cluster_id, doc_ids in clusters.items():
                if len(doc_ids) < 2:
                    continue
                
                # Create pairwise relationships
                for i, doc1 in enumerate(doc_ids):
                    for doc2 in doc_ids[i+1:]:
                        relationship_id = f"cluster_{cluster_id}_{doc1}_{doc2}"
                        
                        relationship_data = {
                            'id': relationship_id,
                            'source_document': doc1,
                            'target_document': doc2,
                            'relationship_type': 'semantic_cluster',
                            'cluster_id': str(cluster_id),
                            'confidence': 0.8  # Default cluster confidence
                        }
                        
                        # Store in relationships collection
                        self.relationships_collection.add(
                            ids=[relationship_id],
                            metadatas=[relationship_data],
                            documents=[f"Cluster relationship: {doc1} <-> {doc2}"]
                        )
                        
        except Exception as e:
            logging.error(f"Error storing cluster relationships: {e}")
    
    def get_related_documents(self, document_id: str, 
                            correlation_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Get documents related to the given document"""
        try:
            # Query relationships where this document is involved
            results = self.relationships_collection.get(
                where={
                    "$or": [
                        {"source_document": document_id},
                        {"target_document": document_id}
                    ]
                },
                include=["metadatas"]
            )
            
            related = []
            for metadata in results.get('metadatas', []):
                confidence = metadata.get('confidence', 0.0)
                if confidence >= correlation_threshold:
                    # Determine the related document ID
                    related_id = (
                        metadata['target_document'] 
                        if metadata['source_document'] == document_id 
                        else metadata['source_document']
                    )
                    
                    related.append({
                        'document_id': related_id,
                        'relationship_type': metadata.get('relationship_type', 'unknown'),
                        'confidence': confidence,
                        'cluster_id': metadata.get('cluster_id')
                    })
            
            # Sort by confidence
            related.sort(key=lambda x: x['confidence'], reverse=True)
            
            return related
            
        except Exception as e:
            logging.error(f"Error getting related documents: {e}")
            return []
    
    def get_cluster_documents(self, cluster_id: str) -> List[str]:
        """Get all documents in a specific cluster"""
        try:
            results = self.relationships_collection.get(
                where={"cluster_id": cluster_id},
                include=["metadatas"]
            )
            
            document_ids = set()
            for metadata in results.get('metadatas', []):
                document_ids.add(metadata['source_document'])
                document_ids.add(metadata['target_document'])
            
            return list(document_ids)
            
        except Exception as e:
            logging.error(f"Error getting cluster documents: {e}")
            return []
    
    def find_document_cluster(self, document_id: str, 
                            correlation_type: str = "semantic") -> List[str]:
        """Find the cluster that contains the given document"""
        try:
            results = self.relationships_collection.get(
                where={
                    "$and": [
                        {
                            "$or": [
                                {"source_document": document_id},
                                {"target_document": document_id}
                            ]
                        },
                        {"relationship_type": f"{correlation_type}_cluster"}
                    ]
                },
                include=["metadatas"]
            )
            
            if not results.get('metadatas'):
                return [document_id]  # Document is in its own cluster
            
            # Get cluster ID from first relationship
            cluster_id = results['metadatas'][0].get('cluster_id')
            if cluster_id:
                return self.get_cluster_documents(cluster_id)
            
            return [document_id]
            
        except Exception as e:
            logging.error(f"Error finding document cluster: {e}")
            return [document_id]
    
    def get_citation_network(self, document_id: str) -> Dict[str, Any]:
        """Get citation network for a document"""
        try:
            # This would require parsing document content for citations
            # For now, return basic structure
            citations = self._extract_citations(document_id)
            references = self._extract_references(document_id)
            
            return {
                'document_id': document_id,
                'citations': citations,
                'references': references,
                'network_depth': min(len(citations) + len(references), self.max_correlation_depth),
                'citation_count': len(citations),
                'reference_count': len(references)
            }
            
        except Exception as e:
            logging.error(f"Error getting citation network: {e}")
            return {
                'document_id': document_id,
                'citations': [],
                'references': [],
                'network_depth': 0,
                'citation_count': 0,
                'reference_count': 0
            }
    
    def _extract_citations(self, document_id: str) -> List[Dict[str, Any]]:
        """Extract citations from document (placeholder implementation)"""
        # This would analyze document content for citation patterns
        # For now, return empty list
        return []
    
    def _extract_references(self, document_id: str) -> List[Dict[str, Any]]:
        """Extract references from document (placeholder implementation)"""
        # This would analyze document content for reference patterns
        # For now, return empty list
        return []
    
    def analyze_dependencies(self, document_ids: List[str]) -> Dict[str, Any]:
        """Analyze dependencies between documents"""
        try:
            dependencies = {}
            dependency_graph = {}
            circular_dependencies = []
            
            for doc_id in document_ids:
                related = self.get_related_documents(doc_id, 0.5)
                dependencies[doc_id] = [r['document_id'] for r in related if r['document_id'] in document_ids]
                dependency_graph[doc_id] = {
                    'depends_on': [r['document_id'] for r in related 
                                 if r['document_id'] in document_ids and r['relationship_type'] == 'dependency'],
                    'depended_by': []
                }
            
            # Build reverse dependencies
            for doc_id, deps in dependency_graph.items():
                for dep_id in deps['depends_on']:
                    if dep_id in dependency_graph:
                        dependency_graph[dep_id]['depended_by'].append(doc_id)
            
            # Detect circular dependencies (simple implementation)
            for doc_id in document_ids:
                visited = set()
                if self._has_circular_dependency(doc_id, dependency_graph, visited):
                    circular_dependencies.append(doc_id)
            
            return {
                'dependencies': dependencies,
                'dependency_graph': dependency_graph,
                'circular_dependencies': list(set(circular_dependencies)),
                'total_relationships': sum(len(deps) for deps in dependencies.values())
            }
            
        except Exception as e:
            logging.error(f"Error analyzing dependencies: {e}")
            return {'dependencies': {}, 'dependency_graph': {}, 'circular_dependencies': []}
    
    def _has_circular_dependency(self, doc_id: str, graph: Dict, 
                               visited: set, depth: int = 0) -> bool:
        """Check for circular dependencies using DFS"""
        if depth > self.max_correlation_depth:
            return False
        
        if doc_id in visited:
            return True
        
        visited.add(doc_id)
        
        for dep_id in graph.get(doc_id, {}).get('depends_on', []):
            if self._has_circular_dependency(dep_id, graph, visited.copy(), depth + 1):
                return True
        
        return False
    
    def validate_relationships(self, correlation_threshold: float = 0.8) -> Dict[str, Any]:
        """Validate cross-document relationships"""
        try:
            all_relationships = self.relationships_collection.get(include=["metadatas"])
            
            valid_relationships = 0
            invalid_relationships = 0
            issues = []
            
            for metadata in all_relationships.get('metadatas', []):
                confidence = metadata.get('confidence', 0.0)
                
                if confidence >= correlation_threshold:
                    valid_relationships += 1
                else:
                    invalid_relationships += 1
                    issues.append(
                        f"Low confidence relationship: {metadata.get('source_document')} -> "
                        f"{metadata.get('target_document')} (confidence: {confidence})"
                    )
            
            validation_status = "good" if invalid_relationships == 0 else "needs_attention"
            if invalid_relationships > valid_relationships:
                validation_status = "poor"
            
            return {
                'validation_status': validation_status,
                'total_relationships': valid_relationships + invalid_relationships,
                'valid_relationships': valid_relationships,
                'invalid_relationships': invalid_relationships,
                'validation_threshold': correlation_threshold,
                'issues': issues[:10]  # Limit issues to first 10
            }
            
        except Exception as e:
            logging.error(f"Error validating relationships: {e}")
            return {
                'validation_status': 'error',
                'issues': [str(e)]
            }
    
    def rebuild_index(self) -> bool:
        """Rebuild the relationship index"""
        try:
            # Delete existing relationships
            self.relationships_collection.delete(
                where={}  # Delete all
            )
            
            # Rebuild would require access to the main documents collection
            # This is a placeholder for the rebuild logic
            logging.info("Relationship index rebuild initiated")
            return True
            
        except Exception as e:
            logging.error(f"Error rebuilding relationship index: {e}")
            return False
    
    def close(self):
        """Clean up relationship manager resources"""
        self.relationships_collection = None
        self.chroma_client = None


class QualityMonitor:
    """Monitors and tracks quality metrics for documents and correlations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
        self._initialize_quality_db()
    
    def _initialize_quality_db(self):
        """Initialize SQLite database for quality tracking"""
        try:
            self.connection = sqlite3.connect(str(self.db_path))
            cursor = self.connection.cursor()
            
            # Create quality tracking tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_quality (
                    document_id TEXT PRIMARY KEY,
                    quality_score REAL,
                    extraction_confidence REAL,
                    processing_time REAL,
                    chunk_count INTEGER,
                    validation_status TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS correlation_quality (
                    correlation_id TEXT PRIMARY KEY,
                    source_document TEXT,
                    target_document TEXT,
                    correlation_score REAL,
                    validation_status TEXT,
                    last_validated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS collection_metrics (
                    metric_name TEXT PRIMARY KEY,
                    metric_value REAL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            self.connection.commit()
            logging.info(f"Quality monitoring database initialized: {self.db_path}")
            
        except Exception as e:
            logging.error(f"Failed to initialize quality database: {e}")
            if self.connection:
                self.connection.close()
                self.connection = None
    
    def update_document_quality(self, document_id: str, quality_metrics: Dict[str, Any]) -> bool:
        """Update quality metrics for a document"""
        if not self.connection:
            return False
        
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO document_quality 
                (document_id, quality_score, extraction_confidence, processing_time, 
                 chunk_count, validation_status, metadata, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                document_id,
                quality_metrics.get('quality_score', 0.0),
                quality_metrics.get('extraction_confidence', 0.0),
                quality_metrics.get('processing_time', 0.0),
                quality_metrics.get('chunk_count', 0),
                quality_metrics.get('validation_status', 'unknown'),
                json.dumps(quality_metrics.get('metadata', {}))
            ))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logging.error(f"Error updating document quality: {e}")
            return False
    
    def get_document_quality(self, document_id: str) -> Dict[str, Any]:
        """Get quality metrics for a specific document"""
        if not self.connection:
            return {'quality_score': 0.0, 'confidence_metrics': {}, 'validation_status': 'unknown'}
        
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                SELECT quality_score, extraction_confidence, processing_time, 
                       chunk_count, validation_status, metadata, last_updated
                FROM document_quality 
                WHERE document_id = ?
            """, (document_id,))
            
            result = cursor.fetchone()
            if not result:
                return {'quality_score': 0.0, 'confidence_metrics': {}, 'validation_status': 'not_found'}
            
            quality_score, extraction_conf, proc_time, chunk_count, val_status, metadata_json, last_updated = result
            
            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
            except:
                metadata = {}
            
            return {
                'quality_score': quality_score,
                'confidence_metrics': {
                    'extraction_confidence': extraction_conf,
                    'processing_time': proc_time,
                    'chunk_count': chunk_count
                },
                'validation_status': val_status,
                'metadata': metadata,
                'last_updated': last_updated
            }
            
        except Exception as e:
            logging.error(f"Error getting document quality: {e}")
            return {'quality_score': 0.0, 'confidence_metrics': {}, 'validation_status': 'error'}
    
    def get_collection_quality(self) -> Dict[str, Any]:
        """Get overall collection quality metrics"""
        if not self.connection:
            return {'quality_score': 0.0, 'document_count': 0, 'avg_metrics': {}}
        
        try:
            cursor = self.connection.cursor()
            
            # Get aggregate quality metrics
            cursor.execute("""
                SELECT 
                    COUNT(*) as doc_count,
                    AVG(quality_score) as avg_quality,
                    AVG(extraction_confidence) as avg_extraction,
                    AVG(processing_time) as avg_proc_time,
                    SUM(chunk_count) as total_chunks
                FROM document_quality
            """)
            
            result = cursor.fetchone()
            if not result or result[0] == 0:
                return {'quality_score': 0.0, 'document_count': 0, 'avg_metrics': {}}
            
            doc_count, avg_quality, avg_extraction, avg_proc_time, total_chunks = result
            
            # Get validation status distribution
            cursor.execute("""
                SELECT validation_status, COUNT(*) 
                FROM document_quality 
                GROUP BY validation_status
            """)
            
            status_distribution = dict(cursor.fetchall())
            
            return {
                'quality_score': avg_quality or 0.0,
                'document_count': doc_count,
                'total_chunks': total_chunks or 0,
                'avg_metrics': {
                    'avg_extraction_confidence': avg_extraction or 0.0,
                    'avg_processing_time': avg_proc_time or 0.0,
                    'avg_chunks_per_document': (total_chunks / doc_count) if doc_count > 0 else 0
                },
                'validation_status_distribution': status_distribution,
                'health_status': self._assess_collection_health(avg_quality, status_distribution)
            }
            
        except Exception as e:
            logging.error(f"Error getting collection quality: {e}")
            return {'quality_score': 0.0, 'document_count': 0, 'avg_metrics': {}}
    
    def _assess_collection_health(self, avg_quality: float, 
                                status_distribution: Dict[str, int]) -> str:
        """Assess overall collection health"""
        if not avg_quality:
            return "unknown"
        
        if avg_quality >= 0.8:
            health = "excellent"
        elif avg_quality >= 0.6:
            health = "good"
        elif avg_quality >= 0.4:
            health = "acceptable"
        else:
            health = "poor"
        
        # Adjust based on validation status
        total_docs = sum(status_distribution.values())
        if total_docs > 0:
            error_rate = status_distribution.get('error', 0) / total_docs
            if error_rate > 0.2:  # More than 20% errors
                health = "poor"
            elif error_rate > 0.1 and health == "excellent":
                health = "good"
        
        return health
    
    def update_collection_metrics(self, metrics: Dict[str, float]) -> bool:
        """Update collection-level metrics"""
        if not self.connection:
            return False
        
        try:
            cursor = self.connection.cursor()
            
            for metric_name, metric_value in metrics.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO collection_metrics 
                    (metric_name, metric_value, last_updated)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (metric_name, metric_value))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logging.error(f"Error updating collection metrics: {e}")
            return False
    
    def get_performance_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get performance trends over time"""
        if not self.connection:
            return {'trends': {}, 'period_days': days}
        
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                SELECT 
                    DATE(last_updated) as date,
                    AVG(quality_score) as avg_quality,
                    COUNT(*) as documents_processed
                FROM document_quality 
                WHERE last_updated >= date('now', '-{} days')
                GROUP BY DATE(last_updated)
                ORDER BY date DESC
            """.format(days))
            
            trends = []
            for row in cursor.fetchall():
                trends.append({
                    'date': row[0],
                    'avg_quality': row[1],
                    'documents_processed': row[2]
                })
            
            return {
                'trends': trends,
                'period_days': days,
                'trend_points': len(trends)
            }
            
        except Exception as e:
            logging.error(f"Error getting performance trends: {e}")
            return {'trends': {}, 'period_days': days}
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None


# Additional FASE 2 helper methods for ChromaVectorStore
class FASE2Helper:
    """Helper methods for FASE 2 cross-document correlation functionality"""
    
    @staticmethod
    def trigger_correlation_update(vector_store, document_id: str, 
                                 correlation_threshold: float = 0.7) -> Dict[str, Any]:
        """Trigger correlation analysis for a newly stored document"""
        if not vector_store.relationship_manager or not vector_store.use_chroma:
            return {'status': 'skipped', 'reason': 'FASE 2 components not available'}
        
        try:
            # Build semantic clusters periodically (every 10 documents)
            doc_count = vector_store.collection.count()
            if doc_count % 10 == 0:  # Rebuild clusters every 10 documents
                cluster_results = vector_store.relationship_manager.build_semantic_clusters(
                    vector_store.collection
                )
                logging.info(f"Built {cluster_results.get('cluster_count', 0)} semantic clusters")
                
                return {
                    'status': 'completed',
                    'action': 'cluster_rebuild',
                    'clusters_built': cluster_results.get('cluster_count', 0),
                    'singleton_count': cluster_results.get('singleton_count', 0)
                }
            
            return {'status': 'skipped', 'reason': 'cluster rebuild not needed'}
            
        except Exception as e:
            logging.error(f"Error triggering correlation update: {e}")
            return {'status': 'error', 'error': str(e)}
    
    @staticmethod
    def optimize_for_scale(vector_store, target_doc_count: int = 10000) -> Dict[str, Any]:
        """Optimize vector store configuration for large scale operations"""
        optimizations = []
        
        try:
            if vector_store.use_chroma:
                current_count = vector_store.collection.count()
                
                # Adjust clustering parameters based on scale
                if vector_store.relationship_manager:
                    if current_count > 1000:
                        vector_store.relationship_manager.similarity_threshold = 0.8  # More strict
                        optimizations.append("Increased similarity threshold for large scale")
                    
                    if current_count > 5000:
                        vector_store.relationship_manager.max_correlation_depth = 2  # Reduce depth
                        optimizations.append("Reduced correlation depth for performance")
                
                # Monitor quality thresholds
                if vector_store.quality_monitor and current_count > 100:
                    quality_stats = vector_store.quality_monitor.get_collection_quality()
                    avg_quality = quality_stats.get('quality_score', 0.0)
                    
                    if avg_quality < 0.6:
                        optimizations.append(f"Quality below threshold: {avg_quality:.3f}")
            
            return {
                'status': 'completed',
                'current_count': current_count if vector_store.use_chroma else 0,
                'target_count': target_doc_count,
                'optimizations_applied': optimizations,
                'performance_level': 'optimal' if len(optimizations) == 0 else 'adjusted'
            }
            
        except Exception as e:
            logging.error(f"Error optimizing for scale: {e}")
            return {'status': 'error', 'error': str(e)}
    
    @staticmethod
    def validate_fase2_integration(vector_store) -> Dict[str, Any]:
        """Validate FASE 2 component integration and functionality"""
        validation_results = {
            'relationship_manager': False,
            'quality_monitor': False,
            'chromadb_available': vector_store.use_chroma,
            'sklearn_available': SKLEARN_AVAILABLE,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Test relationship manager
            if vector_store.relationship_manager:
                try:
                    # Test basic functionality
                    test_relations = vector_store.relationship_manager.get_related_documents('test_doc_id')
                    validation_results['relationship_manager'] = True
                except Exception as e:
                    validation_results['errors'].append(f"Relationship manager error: {e}")
            else:
                validation_results['warnings'].append("Relationship manager not initialized")
            
            # Test quality monitor
            if vector_store.quality_monitor:
                try:
                    # Test basic functionality
                    test_quality = vector_store.quality_monitor.get_collection_quality()
                    validation_results['quality_monitor'] = True
                except Exception as e:
                    validation_results['errors'].append(f"Quality monitor error: {e}")
            else:
                validation_results['warnings'].append("Quality monitor not initialized")
            
            # Check dependencies
            if not SKLEARN_AVAILABLE:
                validation_results['warnings'].append("scikit-learn not available, clustering features limited")
            
            if not vector_store.use_chroma:
                validation_results['warnings'].append("ChromaDB not available, using SQLite fallback")
            
            # Overall status
            if len(validation_results['errors']) == 0:
                if validation_results['relationship_manager'] and validation_results['quality_monitor']:
                    validation_results['status'] = 'fully_integrated'
                elif validation_results['relationship_manager'] or validation_results['quality_monitor']:
                    validation_results['status'] = 'partially_integrated'
                else:
                    validation_results['status'] = 'minimal_integration'
            else:
                validation_results['status'] = 'integration_errors'
            
            return validation_results
            
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {e}")
            validation_results['status'] = 'validation_failed'
            return validation_results