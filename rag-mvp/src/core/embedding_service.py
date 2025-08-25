import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class EmbeddingService:
    """Handles text embedding generation and similarity calculations"""
    
    def __init__(self):
        # Initialize with a simple TF-IDF vectorizer for MVP
        # In production, this could be replaced with sentence-transformers
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        self._is_fitted = False
        self._corpus = []
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text.strip())
        return text.lower()
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        text = self._preprocess_text(text)
        
        if not self._is_fitted:
            # Fit on single text for consistency
            self.vectorizer.fit([text])
            self._is_fitted = True
            self._corpus = [text]
        
        # Transform to vector
        vector = self.vectorizer.transform([text])
        return vector.toarray()[0]  # Return 1D array
    
    def generate_embeddings(self, texts: List[str], skip_empty: bool = False) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        if skip_empty:
            valid_texts = []
            for text in texts:
                try:
                    self._preprocess_text(text)
                    valid_texts.append(text)
                except ValueError:
                    continue
            texts = valid_texts
        else:
            texts = [self._preprocess_text(text) for text in texts]
        
        if not texts:
            return []
        
        # Fit vectorizer on the corpus
        self.vectorizer.fit(texts)
        self._is_fitted = True
        self._corpus = texts.copy()
        
        # Transform all texts
        vectors = self.vectorizer.transform(texts)
        return [vector.toarray()[0] for vector in vectors]
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        # Reshape to 2D arrays for sklearn
        emb1 = embedding1.reshape(1, -1)
        emb2 = embedding2.reshape(1, -1)
        
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)
    
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
    
    def fit_corpus(self, texts: List[str]):
        """Fit the vectorizer on a corpus of texts"""
        clean_texts = [self._preprocess_text(text) for text in texts if text.strip()]
        if clean_texts:
            self.vectorizer.fit(clean_texts)
            self._is_fitted = True
            self._corpus = clean_texts.copy()
    
    def transform_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Transform texts using fitted vectorizer"""
        if not self._is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit_corpus first.")
        
        clean_texts = [self._preprocess_text(text) for text in texts]
        vectors = self.vectorizer.transform(clean_texts)
        return [vector.toarray()[0] for vector in vectors]
    
    def embed_query_and_corpus(self, query_text: str, corpus_texts: List[str]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Embed query and corpus together to ensure same feature space"""
        all_texts = [query_text] + corpus_texts
        all_embeddings = self.generate_embeddings(all_texts)
        
        query_embedding = all_embeddings[0]
        corpus_embeddings = all_embeddings[1:]
        
        return query_embedding, corpus_embeddings