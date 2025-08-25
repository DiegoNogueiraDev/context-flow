import pytest
import numpy as np
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.embedding_service_v2 import EnhancedEmbeddingService, EmbeddingCache


class TestEmbeddingCache:
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def embedding_cache(self, temp_cache_dir):
        """Create EmbeddingCache instance for testing"""
        return EmbeddingCache(temp_cache_dir)
    
    def test_cache_initialization(self, embedding_cache, temp_cache_dir):
        """Test cache initialization"""
        assert embedding_cache.cache_dir.exists()
        assert str(embedding_cache.cache_dir) == temp_cache_dir
    
    def test_cache_key_generation(self, embedding_cache):
        """Test cache key generation"""
        key1 = embedding_cache._get_cache_key("test text", "model1")
        key2 = embedding_cache._get_cache_key("test text", "model2")
        key3 = embedding_cache._get_cache_key("different text", "model1")
        
        assert isinstance(key1, str)
        assert len(key1) == 32  # MD5 hash length
        assert key1 != key2  # Different models should have different keys
        assert key1 != key3  # Different texts should have different keys
    
    def test_cache_set_and_get(self, embedding_cache):
        """Test caching and retrieval of embeddings"""
        text = "test text for caching"
        model = "test-model"
        embedding = np.array([0.1, 0.2, 0.3, 0.4])
        
        # Should return None for non-cached item
        result = embedding_cache.get(text, model)
        assert result is None
        
        # Cache the embedding
        embedding_cache.set(text, model, embedding)
        
        # Should return cached embedding
        result = embedding_cache.get(text, model)
        assert result is not None
        np.testing.assert_array_equal(result, embedding)
    
    def test_cache_clear(self, embedding_cache):
        """Test cache clearing"""
        # Add some items to cache
        embedding_cache.set("text1", "model1", np.array([0.1, 0.2]))
        embedding_cache.set("text2", "model1", np.array([0.3, 0.4]))
        
        # Verify items are cached
        assert embedding_cache.get("text1", "model1") is not None
        assert embedding_cache.get("text2", "model1") is not None
        
        # Clear cache
        embedding_cache.clear()
        
        # Verify items are no longer cached
        assert embedding_cache.get("text1", "model1") is None
        assert embedding_cache.get("text2", "model1") is None


class TestEnhancedEmbeddingService:
    
    @pytest.fixture
    def embedding_service(self):
        """Create EnhancedEmbeddingService instance for testing"""
        # Use fallback mode for testing to avoid downloading models
        return EnhancedEmbeddingService(
            model_name='all-MiniLM-L6-v2',
            use_cache=False,
            fallback_to_tfidf=True
        )
    
    @pytest.fixture
    def embedding_service_with_cache(self):
        """Create EnhancedEmbeddingService with caching for testing"""
        return EnhancedEmbeddingService(
            model_name='all-MiniLM-L6-v2',
            use_cache=True,
            fallback_to_tfidf=True
        )
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing"""
        return [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks with multiple layers",
            "Natural language processing deals with text analysis",
            "Computer vision enables machines to interpret visual information"
        ]
    
    def test_initialization(self, embedding_service):
        """Test enhanced embedding service initialization"""
        assert embedding_service is not None
        assert hasattr(embedding_service, 'model_name')
        assert hasattr(embedding_service, 'use_cache')
        assert hasattr(embedding_service, 'fallback_to_tfidf')
    
    def test_available_models(self):
        """Test available models listing"""
        models = EnhancedEmbeddingService.list_available_models()
        
        assert isinstance(models, dict)
        assert 'all-MiniLM-L6-v2' in models
        assert 'dimensions' in models['all-MiniLM-L6-v2']
        assert 'description' in models['all-MiniLM-L6-v2']
    
    def test_embedding_dimension(self, embedding_service):
        """Test embedding dimension reporting"""
        dim = embedding_service.get_embedding_dimension()
        
        assert isinstance(dim, int)
        assert dim > 0
    
    def test_single_embedding_generation(self, embedding_service):
        """Test generating single embedding"""
        text = "This is a test sentence for embedding generation"
        
        embedding = embedding_service.generate_embedding(text)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1  # 1D vector
        assert embedding.shape[0] > 0  # Has dimensions
        assert not np.isnan(embedding).any()  # No NaN values
        assert not np.isinf(embedding).any()  # No infinite values
    
    def test_batch_embedding_generation(self, embedding_service, sample_texts):
        """Test generating embeddings for multiple texts"""
        embeddings = embedding_service.generate_embeddings(sample_texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(sample_texts)
        
        for embedding in embeddings:
            assert isinstance(embedding, np.ndarray)
            assert len(embedding.shape) == 1
            assert embedding.shape[0] > 0
            assert not np.isnan(embedding).any()
            assert not np.isinf(embedding).any()
        
        # All embeddings should have the same dimension
        dimensions = [len(emb) for emb in embeddings]
        assert len(set(dimensions)) == 1  # All same dimension
    
    def test_embedding_consistency(self, embedding_service):
        """Test that same text produces same embedding"""
        text = "Consistency test sentence"
        
        embedding1 = embedding_service.generate_embedding(text)
        embedding2 = embedding_service.generate_embedding(text)
        
        np.testing.assert_array_equal(embedding1, embedding2)
    
    def test_cosine_similarity(self, embedding_service):
        """Test cosine similarity calculation"""
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
        
        # Self-similarity should be 1.0
        self_similarity = embedding_service.cosine_similarity(emb1, emb1)
        assert abs(self_similarity - 1.0) < 1e-6
    
    def test_find_most_similar(self, embedding_service):
        """Test finding most similar embeddings"""
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
        assert all(isinstance(idx, tuple) for idx in similar_indices)
        assert all(len(idx) == 2 for idx in similar_indices)  # (index, similarity)
        assert all(0 <= idx[0] < len(corpus_texts) for idx in similar_indices)
        
        # Results should be sorted by similarity (descending)
        similarities = [idx[1] for idx in similar_indices]
        assert similarities == sorted(similarities, reverse=True)
    
    def test_embed_query_and_corpus(self, embedding_service, sample_texts):
        """Test embedding query and corpus together"""
        query = "machine learning algorithms"
        
        query_embedding, corpus_embeddings = embedding_service.embed_query_and_corpus(
            query, sample_texts
        )
        
        assert isinstance(query_embedding, np.ndarray)
        assert isinstance(corpus_embeddings, list)
        assert len(corpus_embeddings) == len(sample_texts)
        
        # All embeddings should have the same dimension
        all_embeddings = [query_embedding] + corpus_embeddings
        dimensions = [len(emb) for emb in all_embeddings]
        assert len(set(dimensions)) == 1
    
    def test_empty_text_handling(self, embedding_service):
        """Test handling of empty or whitespace text"""
        with pytest.raises(ValueError):
            embedding_service.generate_embedding("")
        
        with pytest.raises(ValueError):
            embedding_service.generate_embedding("   \n\t  ")
    
    def test_batch_with_empty_texts(self, embedding_service):
        """Test batch processing with skip_empty option"""
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
    
    def test_caching_functionality(self, embedding_service_with_cache):
        """Test embedding caching functionality"""
        text = "This text will be cached"
        
        # Clear cache first
        embedding_service_with_cache.clear_cache()
        
        # First call - should compute and cache
        embedding1 = embedding_service_with_cache.generate_embedding(text)
        
        # Second call - should retrieve from cache
        embedding2 = embedding_service_with_cache.generate_embedding(text)
        
        # Should be identical
        np.testing.assert_array_equal(embedding1, embedding2)
    
    def test_batch_caching(self, embedding_service_with_cache):
        """Test batch embedding with caching"""
        texts = ["text one", "text two", "text three"]
        
        # Clear cache
        embedding_service_with_cache.clear_cache()
        
        # First batch - should cache all
        embeddings1 = embedding_service_with_cache.generate_embeddings(texts)
        
        # Second batch - should retrieve from cache
        embeddings2 = embedding_service_with_cache.generate_embeddings(texts)
        
        # Should be identical
        for emb1, emb2 in zip(embeddings1, embeddings2):
            np.testing.assert_array_equal(emb1, emb2)
    
    def test_model_info(self, embedding_service):
        """Test getting model information"""
        info = embedding_service.get_model_info()
        
        assert isinstance(info, dict)
        assert 'service_type' in info
        assert 'model_name' in info
        assert 'embedding_dimension' in info
        assert info['service_type'] == 'enhanced'
    
    def test_benchmark(self, embedding_service):
        """Test performance benchmarking"""
        sample_texts = [
            "Short text",
            "Medium length text for performance testing",
            "Longer text that contains more words and should take more time to process"
        ]
        
        benchmark_results = embedding_service.benchmark(sample_texts)
        
        assert isinstance(benchmark_results, dict)
        assert 'model_info' in benchmark_results
        assert 'single_embedding_time' in benchmark_results
        assert 'batch_embedding_time' in benchmark_results
        assert 'batch_size' in benchmark_results
        assert 'embedding_dimension' in benchmark_results
        assert 'avg_time_per_text' in benchmark_results
        assert 'self_similarity' in benchmark_results
        
        # Performance metrics should be positive
        assert benchmark_results['single_embedding_time'] > 0
        assert benchmark_results['batch_embedding_time'] > 0
        assert benchmark_results['embedding_dimension'] > 0
        
        # Self-similarity should be very close to 1.0
        assert abs(benchmark_results['self_similarity'] - 1.0) < 0.1
    
    def test_different_batch_sizes(self, embedding_service):
        """Test different batch sizes for embedding generation"""
        texts = [f"Test text number {i}" for i in range(20)]
        
        # Test different batch sizes
        for batch_size in [1, 5, 10, 16, 32]:
            embeddings = embedding_service.generate_embeddings(texts, batch_size=batch_size)
            
            assert len(embeddings) == len(texts)
            assert all(isinstance(emb, np.ndarray) for emb in embeddings)
    
    def test_large_text_handling(self, embedding_service):
        """Test handling of very large texts"""
        # Create a very long text
        large_text = "This is a test sentence. " * 1000  # Very long text
        
        # Should handle gracefully
        embedding = embedding_service.generate_embedding(large_text)
        
        assert isinstance(embedding, np.ndarray)
        assert not np.isnan(embedding).any()
        assert not np.isinf(embedding).any()
    
    def test_special_characters_handling(self, embedding_service):
        """Test handling of special characters and unicode"""
        special_texts = [
            "Text with émojis 🚀 and unicode characters",
            "Mathematical symbols: ∑, ∫, π, α, β",
            "Mixed languages: Hello 世界 мир mundo",
            "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?",
            "\n\t\r Whitespace and newlines \n\t\r"
        ]
        
        for text in special_texts:
            try:
                embedding = embedding_service.generate_embedding(text)
                assert isinstance(embedding, np.ndarray)
                assert not np.isnan(embedding).any()
            except ValueError as e:
                # Only acceptable if text is considered empty after processing
                assert "empty" in str(e).lower()
    
    def test_error_recovery(self, embedding_service):
        """Test error recovery and fallback behavior"""
        # Test with various problematic inputs
        problematic_texts = [
            None,  # Should raise error
            [],    # Should raise error  
            123,   # Should raise error
        ]
        
        for text in problematic_texts:
            with pytest.raises((ValueError, TypeError)):
                embedding_service.generate_embedding(text)
    
    def test_memory_efficiency(self, embedding_service):
        """Test memory efficiency with large batches"""
        # Create a large batch of texts
        large_batch = [f"Document number {i} with some content" for i in range(100)]
        
        # Should handle large batch without memory issues
        embeddings = embedding_service.generate_embeddings(large_batch, batch_size=10)
        
        assert len(embeddings) == 100
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
        
        # Verify embeddings are reasonable
        dimensions = [len(emb) for emb in embeddings]
        assert len(set(dimensions)) == 1  # All same dimension
        
        # Test similarity within batch
        sim = embedding_service.cosine_similarity(embeddings[0], embeddings[1])
        assert 0.0 <= sim <= 1.0