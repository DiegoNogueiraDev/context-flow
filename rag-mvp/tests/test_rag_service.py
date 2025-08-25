import pytest
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.rag_service import RAGService


class TestRAGService:
    
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
    def rag_service(self, temp_db_path):
        """Create RAGService instance for testing"""
        return RAGService(temp_db_path)
    
    def test_initialization(self, rag_service):
        """Test RAG service initialization"""
        assert rag_service is not None
        assert hasattr(rag_service, 'text_processor')
        assert hasattr(rag_service, 'embedding_service')
        assert hasattr(rag_service, 'vector_store')
    
    def test_upload_document_from_text(self, rag_service):
        """Test uploading and processing a document from text"""
        content = """
        Machine learning is a subset of artificial intelligence that enables computers to learn 
        and improve from experience without being explicitly programmed. It focuses on the 
        development of computer programs that can access data and use it to learn for themselves.
        
        Deep learning is a subset of machine learning that uses neural networks with multiple layers.
        These networks can automatically discover patterns in data without manual feature extraction.
        """
        filename = "ml_guide.txt"
        
        document_id = rag_service.upload_document(content, filename)
        
        assert document_id is not None
        assert isinstance(document_id, str)
        assert len(document_id) > 0
        
        # Verify document was stored
        documents = rag_service.get_all_documents()
        assert len(documents) == 1
        assert documents[0]['filename'] == filename
    
    def test_upload_document_from_bytes(self, rag_service):
        """Test uploading document from bytes (simulating file upload)"""
        content_bytes = b"This is file content from uploaded bytes."
        filename = "upload.txt"
        
        document_id = rag_service.upload_document_bytes(content_bytes, filename)
        
        assert document_id is not None
        documents = rag_service.get_all_documents()
        assert len(documents) == 1
        assert "uploaded bytes" in documents[0]['content']
    
    def test_search_documents_semantic(self, rag_service):
        """Test semantic search across uploaded documents"""
        # Upload multiple documents
        doc1_content = "Machine learning algorithms can learn from training data to make predictions."
        doc2_content = "Natural language processing helps computers understand human language."
        doc3_content = "Computer vision enables machines to interpret and understand visual information."
        
        rag_service.upload_document(doc1_content, "ml.txt")
        rag_service.upload_document(doc2_content, "nlp.txt")
        rag_service.upload_document(doc3_content, "cv.txt")
        
        # Search for ML-related content
        results = rag_service.search("machine learning and algorithms", top_k=5)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all('content' in result for result in results)
        assert all('similarity' in result for result in results)
        assert all('filename' in result for result in results)
        
        # First result should be most relevant (ML document)
        top_result = results[0]
        assert 'machine learning' in top_result['content'].lower() or 'algorithms' in top_result['content'].lower()
    
    def test_search_returns_ranked_results(self, rag_service):
        """Test that search results are properly ranked by relevance"""
        # Upload documents with varying relevance
        highly_relevant = "Artificial intelligence and machine learning are transforming technology."
        somewhat_relevant = "Technology is advancing rapidly in many fields including AI."
        not_relevant = "Cooking pasta requires boiling water and adding salt."
        
        rag_service.upload_document(highly_relevant, "ai_article.txt")
        rag_service.upload_document(somewhat_relevant, "tech_news.txt")
        rag_service.upload_document(not_relevant, "cooking.txt")
        
        results = rag_service.search("artificial intelligence machine learning", top_k=3)
        
        assert len(results) == 3
        
        # Results should be in descending order of similarity
        similarities = [result['similarity'] for result in results]
        assert similarities == sorted(similarities, reverse=True)
        
        # Most relevant should be first
        assert 'artificial intelligence' in results[0]['content'].lower()
    
    def test_search_with_question(self, rag_service):
        """Test searching with a question format"""
        content = """
        What is machine learning? Machine learning is a method of data analysis that automates 
        analytical model building. It is a branch of artificial intelligence based on the idea 
        that systems can learn from data, identify patterns and make decisions with minimal human intervention.
        
        How does deep learning work? Deep learning uses artificial neural networks with multiple layers 
        to model and understand complex patterns in datasets.
        """
        
        rag_service.upload_document(content, "ml_faq.txt")
        
        results = rag_service.search("What is machine learning?", top_k=3)
        
        assert len(results) > 0
        # Should find relevant content
        top_result = results[0]
        assert any(keyword in top_result['content'].lower() for keyword in ['machine learning', 'data analysis', 'artificial intelligence'])
    
    def test_get_document_details(self, rag_service):
        """Test retrieving document details with chunks"""
        content = "This is a test document with multiple sentences. Each sentence will be chunked separately. This helps with semantic search."
        filename = "test_doc.txt"
        
        doc_id = rag_service.upload_document(content, filename)
        details = rag_service.get_document_details(doc_id)
        
        assert details is not None
        assert details['id'] == doc_id
        assert details['filename'] == filename
        assert 'chunks' in details
        assert isinstance(details['chunks'], list)
        assert len(details['chunks']) > 0
    
    def test_delete_document(self, rag_service):
        """Test deleting a document"""
        content = "Document to be deleted."
        filename = "temp.txt"
        
        doc_id = rag_service.upload_document(content, filename)
        
        # Verify document exists
        docs_before = rag_service.get_all_documents()
        assert len(docs_before) == 1
        
        # Delete document
        result = rag_service.delete_document(doc_id)
        assert result is True
        
        # Verify deletion
        docs_after = rag_service.get_all_documents()
        assert len(docs_after) == 0
    
    def test_empty_search(self, rag_service):
        """Test search with no documents uploaded"""
        results = rag_service.search("any query", top_k=5)
        assert results == []
    
    def test_search_with_empty_query(self, rag_service):
        """Test search with empty query"""
        rag_service.upload_document("Some content", "file.txt")
        
        with pytest.raises(ValueError):
            rag_service.search("", top_k=5)
        
        with pytest.raises(ValueError):
            rag_service.search("   ", top_k=5)
    
    def test_chunk_correlation_across_documents(self, rag_service):
        """Test that search can correlate knowledge across different documents"""
        # Upload related documents
        doc1 = "Python is a programming language used for machine learning."
        doc2 = "Scikit-learn is a popular machine learning library for Python."
        doc3 = "TensorFlow is another framework for deep learning in Python."
        
        rag_service.upload_document(doc1, "python_ml.txt")
        rag_service.upload_document(doc2, "sklearn.txt")
        rag_service.upload_document(doc3, "tensorflow.txt")
        
        # Search should find relevant chunks across all documents
        results = rag_service.search("Python machine learning libraries", top_k=5)
        
        # Should get results from multiple documents
        filenames = [result['filename'] for result in results]
        unique_files = set(filenames)
        assert len(unique_files) > 1  # Multiple files should be represented
        
        # All results should be relevant to the query
        for result in results:
            content_lower = result['content'].lower()
            assert any(keyword in content_lower for keyword in ['python', 'machine learning', 'library', 'framework'])
    
    def test_performance_with_multiple_documents(self, rag_service):
        """Test performance with multiple documents"""
        # Upload several documents
        for i in range(10):
            content = f"This is document {i} about topic {i % 3}. " * 10
            rag_service.upload_document(content, f"doc_{i}.txt")
        
        # Search should still work efficiently
        results = rag_service.search("document topic", top_k=3)
        
        assert len(results) == 3
        assert all('similarity' in result for result in results)