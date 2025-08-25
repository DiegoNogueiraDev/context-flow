import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.embedding_service import EmbeddingService
from core.models import Chunk


class TestEmbeddingService:
    
    @pytest.fixture
    def embedding_service(self):
        """Create EmbeddingService instance for testing"""
        return EmbeddingService()
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing"""
        return [
            Chunk("Machine learning is a subset of artificial intelligence"),
            Chunk("Deep learning uses neural networks with multiple layers"),
            Chunk("Python is a popular programming language"),
            Chunk("Natural language processing deals with text analysis")
        ]
    
    def test_generate_single_embedding(self, embedding_service):
        """Test generating embedding for single text"""
        text = "This is a test sentence for embedding generation"
        
        embedding = embedding_service.generate_embedding(text)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1  # 1D vector
        assert embedding.shape[0] > 0  # Has dimensions
        assert not np.isnan(embedding).any()  # No NaN values
        assert not np.isinf(embedding).any()  # No infinite values
    
    def test_generate_batch_embeddings(self, embedding_service, sample_chunks):
        """Test generating embeddings for multiple chunks"""
        texts = [chunk.content for chunk in sample_chunks]
        
        embeddings = embedding_service.generate_embeddings(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        
        for embedding in embeddings:
            assert isinstance(embedding, np.ndarray)
            assert len(embedding.shape) == 1
            assert embedding.shape[0] > 0
            assert not np.isnan(embedding).any()
            assert not np.isinf(embedding).any()
    
    def test_embedding_consistency(self, embedding_service):
        """Test that same text produces same embedding"""
        text = "Consistency test sentence"
        
        embedding1 = embedding_service.generate_embedding(text)
        embedding2 = embedding_service.generate_embedding(text)
        
        np.testing.assert_array_equal(embedding1, embedding2)
    
    def test_similarity_calculation(self, embedding_service):
        """Test cosine similarity calculation between embeddings"""
        text1 = "Machine learning and artificial intelligence"
        text2 = "AI and machine learning technologies"
        text3 = "Cooking recipes and kitchen tools"
        
        emb1 = embedding_service.generate_embedding(text1)
        emb2 = embedding_service.generate_embedding(text2)
        emb3 = embedding_service.generate_embedding(text3)
        
        # Similar texts should have higher similarity
        similarity_12 = embedding_service.cosine_similarity(emb1, emb2)
        similarity_13 = embedding_service.cosine_similarity(emb1, emb3)
        
        assert isinstance(similarity_12, float)
        assert isinstance(similarity_13, float)
        assert -1.0 <= similarity_12 <= 1.0
        assert -1.0 <= similarity_13 <= 1.0
        assert similarity_12 > similarity_13  # Related texts more similar
    
    def test_find_similar_embeddings(self, embedding_service):
        """Test finding most similar embeddings from a collection"""
        query_text = "artificial intelligence and machine learning"
        corpus_texts = [
            "Machine learning is part of AI field",
            "Cooking pasta with tomato sauce", 
            "Deep learning neural networks",
            "Weather forecast for tomorrow",
            "Natural language processing in AI"
        ]
        
        query_embedding, corpus_embeddings = embedding_service.embed_query_and_corpus(
            query_text, corpus_texts
        )
        
        similar_indices = embedding_service.find_most_similar(
            query_embedding, corpus_embeddings, top_k=3
        )
        
        assert isinstance(similar_indices, list)
        assert len(similar_indices) == 3
        assert all(isinstance(idx, tuple) for idx in similar_indices)  # (index, similarity)
        assert all(0 <= idx[0] < len(corpus_texts) for idx in similar_indices)
        
        # Should return AI/ML related texts first
        top_index = similar_indices[0][0]
        assert any(keyword in corpus_texts[top_index].lower() 
                  for keyword in ['machine', 'learning', 'ai', 'neural', 'processing'])
    
    def test_empty_text_handling(self, embedding_service):
        """Test handling of empty or whitespace text"""
        with pytest.raises(ValueError):
            embedding_service.generate_embedding("")
            
        with pytest.raises(ValueError):
            embedding_service.generate_embedding("   \n\t  ")
    
    def test_batch_embedding_with_empty_texts(self, embedding_service):
        """Test batch processing filters out empty texts"""
        texts = [
            "Valid text one",
            "",
            "Valid text two",
            "   ",
            "Valid text three"
        ]
        
        embeddings = embedding_service.generate_embeddings(texts, skip_empty=True)
        
        # Should only return embeddings for non-empty texts
        assert len(embeddings) == 3
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)