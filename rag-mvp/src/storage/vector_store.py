import sqlite3
import json
import numpy as np
from typing import List, Dict, Any, Optional
from core.models import Document


class VectorStore:
    """Handles storage and retrieval of documents, chunks, and embeddings"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
        return self.conn
    
    def initialize_database(self):
        """Initialize database tables"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Chunks table with embeddings stored as JSON
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                content TEXT NOT NULL,
                start_index INTEGER DEFAULT 0,
                end_index INTEGER DEFAULT 0,
                embedding TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            )
        ''')
        
        # Create index for faster searches
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id 
            ON chunks (document_id)
        ''')
        
        conn.commit()
    
    def _embedding_to_json(self, embedding: np.ndarray) -> str:
        """Convert numpy array to JSON string"""
        return json.dumps(embedding.tolist())
    
    def _json_to_embedding(self, json_str: str) -> np.ndarray:
        """Convert JSON string to numpy array"""
        return np.array(json.loads(json_str))
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
    
    def store_document(self, document: Document, embeddings: List[np.ndarray]) -> bool:
        """Store document with its chunks and embeddings"""
        if len(embeddings) != len(document.chunks):
            raise ValueError(f"Number of embeddings ({len(embeddings)}) must match number of chunks ({len(document.chunks)})")
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Store document
            cursor.execute('''
                INSERT OR REPLACE INTO documents (id, filename, content)
                VALUES (?, ?, ?)
            ''', (document.id, document.filename, document.content))
            
            # Delete existing chunks for this document
            cursor.execute('DELETE FROM chunks WHERE document_id = ?', (document.id,))
            
            # Store chunks with embeddings
            for chunk, embedding in zip(document.chunks, embeddings):
                cursor.execute('''
                    INSERT INTO chunks (id, document_id, content, start_index, end_index, embedding)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    chunk.chunk_id,
                    document.id,
                    chunk.content,
                    chunk.start_index,
                    chunk.end_index,
                    self._embedding_to_json(embedding)
                ))
            
            conn.commit()
            return True
            
        except Exception as e:
            conn.rollback()
            raise e
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using cosine similarity"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get all chunks with embeddings
        cursor.execute('''
            SELECT c.id as chunk_id, c.document_id, c.content, c.embedding,
                   d.filename
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
        ''')
        
        results = []
        for row in cursor.fetchall():
            stored_embedding = self._json_to_embedding(row['embedding'])
            similarity = self._cosine_similarity(query_embedding, stored_embedding)
            
            results.append({
                'chunk_id': row['chunk_id'],
                'document_id': row['document_id'],
                'content': row['content'],
                'filename': row['filename'],
                'similarity': similarity
            })
        
        # Sort by similarity (descending) and return top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id as chunk_id, content, start_index, end_index
            FROM chunks
            WHERE document_id = ?
            ORDER BY start_index
        ''', (document_id,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all stored documents"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, content, created_at
            FROM documents
            ORDER BY created_at DESC
        ''')
        
        return [dict(row) for row in cursor.fetchall()]
    
    def delete_document(self, document_id: str) -> bool:
        """Delete document and all its chunks"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Delete chunks (will cascade due to foreign key)
            cursor.execute('DELETE FROM chunks WHERE document_id = ?', (document_id,))
            
            # Delete document
            cursor.execute('DELETE FROM documents WHERE id = ?', (document_id,))
            
            conn.commit()
            return cursor.rowcount > 0
            
        except Exception:
            conn.rollback()
            return False
    
    def update_document(self, document: Document, embeddings: List[np.ndarray]) -> bool:
        """Update existing document"""
        # For simplicity, update is same as store (replace)
        return self.store_document(document, embeddings)
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None