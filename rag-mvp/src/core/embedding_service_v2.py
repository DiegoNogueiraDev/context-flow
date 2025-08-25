"""
Enhanced embedding service using sentence-transformers for better semantic understanding.
Maintains backward compatibility with the original TF-IDF based service.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
import hashlib
import os
import pickle
from pathlib import Path
import time

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Falling back to TF-IDF.")

from .embedding_service import EmbeddingService as TFIDFEmbeddingService


class EmbeddingCache:
    """Simple file-based cache for embeddings"""
    
    def __init__(self, cache_dir: str = ".embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key from text and model"""
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Get cached embedding"""
        key = self._get_cache_key(text, model_name)
        cache_file = self.cache_dir / f"{key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                # Invalid cache file, remove it
                cache_file.unlink()
        
        return None
    
    def set(self, text: str, model_name: str, embedding: np.ndarray):
        """Cache embedding"""
        try:
            key = self._get_cache_key(text, model_name)
            cache_file = self.cache_dir / f"{key}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logging.warning(f"Failed to cache embedding: {e}")
    
    def clear(self):
        """Clear all cached embeddings"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()


class EnhancedEmbeddingService:
    """Enhanced embedding service with sentence-transformers support"""
    
    # Available models (ordered by quality/size trade-off)
    AVAILABLE_MODELS = {
        'all-MiniLM-L6-v2': {
            'dimensions': 384,
            'size': '80MB',
            'speed': 'fast',
            'quality': 'good',
            'description': 'Balanced model for most use cases'
        },
        'all-mpnet-base-v2': {
            'dimensions': 768,
            'size': '420MB', 
            'speed': 'medium',
            'quality': 'excellent',
            'description': 'Best quality, larger size'
        },
        'paraphrase-multilingual-MiniLM-L12-v2': {
            'dimensions': 384,
            'size': '110MB',
            'speed': 'fast',
            'quality': 'good',
            'description': 'Supports multiple languages'
        }
    }
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 use_cache: bool = True,
                 fallback_to_tfidf: bool = True,
                 device: str = 'cpu'):
        """
        Initialize enhanced embedding service
        
        Args:
            model_name: Name of sentence-transformer model
            use_cache: Whether to cache embeddings
            fallback_to_tfidf: Whether to fallback to TF-IDF if sentence-transformers unavailable
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.use_cache = use_cache
        self.fallback_to_tfidf = fallback_to_tfidf
        self.device = device
        
        # Initialize cache
        if use_cache:
            self.cache = EmbeddingCache()
        
        # Try to initialize sentence transformer model
        self.model = None
        self.model_info = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logging.info(f"Loading sentence-transformer model: {model_name}")
                self.model = SentenceTransformer(model_name, device=device)
                self.model_info = self.AVAILABLE_MODELS.get(model_name, {})
                logging.info(f"Model loaded successfully. Dimensions: {self.get_embedding_dimension()}")
            except Exception as e:
                logging.error(f"Failed to load sentence-transformer model: {e}")
                self.model = None
        
        # Fallback to TF-IDF if needed
        if self.model is None and fallback_to_tfidf:
            logging.info("Falling back to TF-IDF embedding service")
            self.tfidf_service = TFIDFEmbeddingService()
        elif self.model is None:
            raise ValueError("No embedding service available")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this service"""
        if self.model:
            # Get dimension from model
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                return self.model.get_sentence_embedding_dimension()
            else:
                # Fallback: encode a test string to get dimension
                test_embedding = self.model.encode("test")
                return len(test_embedding)
        else:
            # TF-IDF fallback - dimension varies, return default
            return 1000
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        
        text = text.strip()
        
        # Try cache first
        if self.use_cache and self.cache:
            cached = self.cache.get(text, self.model_name)
            if cached is not None:
                return cached
        
        # Generate embedding
        if self.model:
            embedding = self._generate_with_sentence_transformer(text)
        else:
            embedding = self._generate_with_tfidf(text)
        
        # Cache result
        if self.use_cache and self.cache:
            self.cache.set(text, self.model_name, embedding)
        
        return embedding
    
    def generate_embeddings(self, texts: List[str], 
                          skip_empty: bool = False,
                          batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings for multiple texts with batching"""
        if skip_empty:
            valid_texts = [text.strip() for text in texts if text and text.strip()]
        else:
            valid_texts = [text.strip() for text in texts]
            # Check for empty texts
            for text in valid_texts:
                if not text:
                    raise ValueError("Text cannot be empty or whitespace only")
        
        if not valid_texts:
            return []
        
        embeddings = []
        
        # Check cache first
        if self.use_cache and self.cache:
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(valid_texts):
                cached = self.cache.get(text, self.model_name)
                if cached is not None:
                    cached_embeddings.append((i, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                if self.model:
                    new_embeddings = self._generate_batch_with_sentence_transformer(
                        uncached_texts, batch_size
                    )
                else:
                    new_embeddings = [self._generate_with_tfidf(text) for text in uncached_texts]
                
                # Cache new embeddings
                for text, embedding in zip(uncached_texts, new_embeddings):
                    self.cache.set(text, self.model_name, embedding)
            else:
                new_embeddings = []
            
            # Combine cached and new embeddings in correct order
            all_embeddings = [None] * len(valid_texts)
            
            # Place cached embeddings
            for i, embedding in cached_embeddings:
                all_embeddings[i] = embedding
            
            # Place new embeddings
            for i, embedding in zip(uncached_indices, new_embeddings):
                all_embeddings[i] = embedding
            
            embeddings = all_embeddings
        
        else:
            # No cache, generate all embeddings
            if self.model:
                embeddings = self._generate_batch_with_sentence_transformer(valid_texts, batch_size)
            else:
                embeddings = [self._generate_with_tfidf(text) for text in valid_texts]
        
        return embeddings
    
    def _generate_with_sentence_transformer(self, text: str) -> np.ndarray:
        """Generate embedding using sentence transformer"""
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logging.error(f"Sentence transformer encoding failed: {e}")
            if hasattr(self, 'tfidf_service'):
                return self._generate_with_tfidf(text)
            else:
                raise
    
    def _generate_batch_with_sentence_transformer(self, texts: List[str], batch_size: int) -> List[np.ndarray]:
        """Generate embeddings in batches using sentence transformer"""
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 100  # Show progress for large batches
            )
            return [emb.astype(np.float32) for emb in embeddings]
        except Exception as e:
            logging.error(f"Batch sentence transformer encoding failed: {e}")
            if hasattr(self, 'tfidf_service'):
                return [self._generate_with_tfidf(text) for text in texts]
            else:
                raise
    
    def _generate_with_tfidf(self, text: str) -> np.ndarray:
        """Generate embedding using TF-IDF fallback"""
        return self.tfidf_service.generate_embedding(text)
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        # Ensure embeddings are numpy arrays
        emb1 = np.array(embedding1).flatten()
        emb2 = np.array(embedding2).flatten()
        
        # Calculate cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm_a = np.linalg.norm(emb1)
        norm_b = np.linalg.norm(emb2)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         corpus_embeddings: List[np.ndarray], 
                         top_k: int = 5) -> List[Tuple[int, float]]:
        """Find most similar embeddings from corpus"""
        similarities = []
        
        for i, corpus_emb in enumerate(corpus_embeddings):
            similarity = self.cosine_similarity(query_embedding, corpus_emb)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def embed_query_and_corpus(self, query_text: str, corpus_texts: List[str]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Embed query and corpus together for consistency"""
        all_texts = [query_text] + corpus_texts
        all_embeddings = self.generate_embeddings(all_texts)
        
        query_embedding = all_embeddings[0]
        corpus_embeddings = all_embeddings[1:]
        
        return query_embedding, corpus_embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        info = {
            'service_type': 'enhanced',
            'model_name': self.model_name,
            'embedding_dimension': self.get_embedding_dimension(),
            'uses_cache': self.use_cache,
            'device': self.device
        }
        
        if self.model_info:
            info.update(self.model_info)
        
        if not self.model:
            info['fallback_mode'] = 'tfidf'
        
        return info
    
    def benchmark(self, sample_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Benchmark the embedding service performance"""
        if sample_texts is None:
            sample_texts = [
                "This is a short test sentence.",
                "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
                "The quick brown fox jumps over the lazy dog. This is a longer sentence to test processing time.",
                "Natural language processing involves computational techniques for analyzing and synthesizing human language.",
                "Deep learning uses neural networks with multiple layers to model complex patterns in data."
            ]
        
        # Time single embedding
        start_time = time.time()
        single_embedding = self.generate_embedding(sample_texts[0])
        single_time = time.time() - start_time
        
        # Time batch embedding
        start_time = time.time()
        batch_embeddings = self.generate_embeddings(sample_texts)
        batch_time = time.time() - start_time
        
        # Calculate similarity
        similarity = self.cosine_similarity(single_embedding, batch_embeddings[0])
        
        return {
            'model_info': self.get_model_info(),
            'single_embedding_time': single_time,
            'batch_embedding_time': batch_time,
            'batch_size': len(sample_texts),
            'embedding_dimension': len(single_embedding),
            'avg_time_per_text': batch_time / len(sample_texts),
            'self_similarity': similarity  # Should be 1.0
        }
    
    def clear_cache(self):
        """Clear embedding cache"""
        if self.use_cache and self.cache:
            self.cache.clear()
            logging.info("Embedding cache cleared")
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """List all available sentence transformer models"""
        return cls.AVAILABLE_MODELS