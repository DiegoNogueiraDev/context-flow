import pytest
import numpy as np
import sqlite3
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from storage.vector_store import VectorStore
from core.models import Document, Chunk


class TestVectorStore:
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database file"""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.unlink(path)
    
    @pytest.fixture
    def vector_store(self, temp_db_path):
        """Create VectorStore instance for testing"""
        return VectorStore(temp_db_path)
    
    @pytest.fixture
    def sample_document(self):
        """Sample document with chunks and embeddings"""
        doc = Document(
            content="This is a sample document about machine learning and AI.",
            filename="sample.txt"
        )
        doc.chunks = [
            Chunk("This is a sample document about machine learning"),
            Chunk("machine learning and AI are related fields"),
            Chunk("AI applications are growing rapidly")
        ]
        return doc
    
    @pytest.fixture
    def sample_embeddings(self):
        """Sample embeddings matching the chunks"""
        return [
            np.random.rand(100).astype(np.float32),
            np.random.rand(100).astype(np.float32),
            np.random.rand(100).astype(np.float32)
        ]
    
    def test_initialize_database(self, vector_store, temp_db_path):
        """Test database initialization creates required tables"""
        vector_store.initialize_database()
        
        # Check if database file exists
        assert os.path.exists(temp_db_path)
        
        # Check if tables were created
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        
        # Check documents table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
        assert cursor.fetchone() is not None
        
        # Check chunks table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'")
        assert cursor.fetchone() is not None
        
        conn.close()
    
    def test_store_document(self, vector_store, sample_document, sample_embeddings):
        """Test storing a document with chunks and embeddings"""
        vector_store.initialize_database()
        
        result = vector_store.store_document(sample_document, sample_embeddings)
        
        assert result is True
        
        # Verify document was stored
        stored_docs = vector_store.get_all_documents()
        assert len(stored_docs) == 1
        assert stored_docs[0]['filename'] == sample_document.filename
        
        # Verify chunks were stored
        stored_chunks = vector_store.get_document_chunks(sample_document.id)
        assert len(stored_chunks) == len(sample_document.chunks)
    
    def test_store_document_validation(self, vector_store, sample_document):
        """Test validation when storing documents"""
        vector_store.initialize_database()
        
        # Test with mismatched embeddings count
        wrong_embeddings = [np.random.rand(100).astype(np.float32)]  # Only 1 embedding for 3 chunks
        
        with pytest.raises(ValueError):
            vector_store.store_document(sample_document, wrong_embeddings)
    
    def test_search_similar_chunks(self, vector_store, sample_document, sample_embeddings):
        """Test searching for similar chunks"""
        vector_store.initialize_database()
        vector_store.store_document(sample_document, sample_embeddings)
        
        # Create query embedding
        query_embedding = np.random.rand(100).astype(np.float32)
        
        results = vector_store.search_similar(query_embedding, top_k=2)
        
        assert isinstance(results, list)
        assert len(results) <= 2  # Should return at most top_k results
        
        # Each result should have required fields
        for result in results:
            assert 'chunk_id' in result
            assert 'content' in result
            assert 'similarity' in result
            assert 'document_id' in result
            assert isinstance(result['similarity'], float)
    
    def test_search_empty_database(self, vector_store):
        """Test searching in empty database"""
        vector_store.initialize_database()
        
        query_embedding = np.random.rand(100).astype(np.float32)
        results = vector_store.search_similar(query_embedding, top_k=5)
        
        assert results == []
    
    def test_get_document_chunks(self, vector_store, sample_document, sample_embeddings):
        """Test retrieving chunks for a specific document"""
        vector_store.initialize_database()
        vector_store.store_document(sample_document, sample_embeddings)
        
        chunks = vector_store.get_document_chunks(sample_document.id)
        
        assert len(chunks) == len(sample_document.chunks)
        assert all('chunk_id' in chunk for chunk in chunks)
        assert all('content' in chunk for chunk in chunks)
        assert all('start_index' in chunk for chunk in chunks)
    
    def test_delete_document(self, vector_store, sample_document, sample_embeddings):
        """Test deleting a document and its chunks"""
        vector_store.initialize_database()
        vector_store.store_document(sample_document, sample_embeddings)
        
        # Verify document exists
        docs_before = vector_store.get_all_documents()
        assert len(docs_before) == 1
        
        # Delete document
        result = vector_store.delete_document(sample_document.id)
        assert result is True
        
        # Verify document and chunks are deleted
        docs_after = vector_store.get_all_documents()
        assert len(docs_after) == 0
        
        chunks_after = vector_store.get_document_chunks(sample_document.id)
        assert len(chunks_after) == 0
    
    def test_get_all_documents(self, vector_store, sample_embeddings):
        """Test retrieving all documents"""
        vector_store.initialize_database()
        
        # Store multiple documents
        doc1 = Document("Content 1", "file1.txt")
        doc1.chunks = [Chunk("Chunk 1")]
        doc2 = Document("Content 2", "file2.txt")  
        doc2.chunks = [Chunk("Chunk 2")]
        
        vector_store.store_document(doc1, [sample_embeddings[0]])
        vector_store.store_document(doc2, [sample_embeddings[1]])
        
        all_docs = vector_store.get_all_documents()
        
        assert len(all_docs) == 2
        filenames = [doc['filename'] for doc in all_docs]
        assert 'file1.txt' in filenames
        assert 'file2.txt' in filenames
    
    def test_update_document(self, vector_store, sample_document, sample_embeddings):
        """Test updating an existing document"""
        vector_store.initialize_database()
        vector_store.store_document(sample_document, sample_embeddings)
        
        # Update document content
        updated_content = "This is updated content about AI and ML."
        updated_doc = Document(updated_content, sample_document.filename)
        updated_doc.id = sample_document.id  # Same ID
        updated_doc.chunks = [Chunk("Updated chunk about AI and ML")]
        
        result = vector_store.update_document(updated_doc, [sample_embeddings[0]])
        assert result is True
        
        # Verify update
        stored_docs = vector_store.get_all_documents()
        assert len(stored_docs) == 1
        assert stored_docs[0]['content'] == updated_content
        
        # Should have new chunks
        chunks = vector_store.get_document_chunks(updated_doc.id)
        assert len(chunks) == 1
        assert "Updated chunk" in chunks[0]['content']
    
    def test_database_persistence(self, temp_db_path, sample_document, sample_embeddings):
        """Test that data persists across VectorStore instances"""
        # Create first instance and store data
        store1 = VectorStore(temp_db_path)
        store1.initialize_database()
        store1.store_document(sample_document, sample_embeddings)
        
        # Create second instance and verify data exists
        store2 = VectorStore(temp_db_path)
        docs = store2.get_all_documents()
        
        assert len(docs) == 1
        assert docs[0]['filename'] == sample_document.filename