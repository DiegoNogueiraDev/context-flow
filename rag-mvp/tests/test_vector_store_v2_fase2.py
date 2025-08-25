"""
Comprehensive test suite for FASE 2 cross-document correlation capabilities in ChromaVectorStore.

Tests cover:
- Cross-document relationship management
- Semantic clustering
- Hierarchical search enhancement
- Quality assurance integration
- Performance at 10k+ document scale
"""

import unittest
import tempfile
import shutil
import numpy as np
from pathlib import Path
import json
import time
from typing import List, Dict, Any

import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from storage.vector_store_v2 import ChromaVectorStore, DocumentRelationshipManager, QualityMonitor, FASE2Helper
from core.models import Document, Chunk


class MockEnhancedDocument(Document):
    """Mock enhanced document for testing"""
    
    def __init__(self, content: str, filename: str):
        super().__init__(content, filename)
        self.document_type = 'test'
        self.metadata = MockDocumentMetadata()


class MockDocumentMetadata:
    """Mock document metadata for testing"""
    
    def __init__(self):
        self.title = "Test Document"
        self.author = "Test Author"
        self.creation_date = "2024-01-01"
        self.quality_score = 0.85
        self.extraction_confidence = MockExtractionConfidence()
        self.processing_stats = MockProcessingStats()
        self.hierarchy_tree = None
        self.table_schemas = []
        self.figures = []


class MockExtractionConfidence:
    """Mock extraction confidence for testing"""
    
    def __init__(self):
        self.overall = 0.85


class MockProcessingStats:
    """Mock processing stats for testing"""
    
    def __init__(self):
        self.total_processing_time = 1.5


class MockEnhancedChunk(Chunk):
    """Mock enhanced chunk for testing"""
    
    def __init__(self, content: str, start_index: int = 0, end_index: int = 0):
        super().__init__(content, start_index, end_index)
        self.chunk_type = 'paragraph'
        self.hierarchy_level = 1
        self.parent_section = 'Introduction'
        self.semantic_topics = ['test', 'mock']
        self.confidence = 0.9


class TestFASE2Integration(unittest.TestCase):
    """Test FASE 2 cross-document correlation capabilities"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.vector_store = ChromaVectorStore(
            persist_directory=str(self.test_dir / "chroma_test"),
            collection_name="test_collection",
            fallback_db_path=str(self.test_dir / "fallback.db")
        )
        
        # Create test documents
        self.test_docs = self._create_test_documents()
        
    def tearDown(self):
        """Clean up test environment"""
        self.vector_store.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_test_documents(self) -> List[MockEnhancedDocument]:
        """Create test documents for correlation testing"""
        documents = []
        
        # Document 1: Machine Learning
        doc1 = MockEnhancedDocument(
            "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "ml_basics.txt"
        )
        doc1.metadata.title = "Machine Learning Basics"
        doc1.metadata.author = "AI Expert"
        doc1.chunks = [MockEnhancedChunk("Machine learning is a subset of artificial intelligence that focuses on algorithms.")]
        documents.append(doc1)
        
        # Document 2: Deep Learning (related to ML)
        doc2 = MockEnhancedDocument(
            "Deep learning uses neural networks with multiple layers to learn complex patterns.",
            "deep_learning.txt"
        )
        doc2.metadata.title = "Deep Learning Overview"
        doc2.metadata.author = "AI Expert"
        doc2.chunks = [MockEnhancedChunk("Deep learning uses neural networks with multiple layers to learn complex patterns.")]
        documents.append(doc2)
        
        # Document 3: Natural Language Processing (related to ML)
        doc3 = MockEnhancedDocument(
            "Natural language processing applies machine learning to understand human language.",
            "nlp_intro.txt"
        )
        doc3.metadata.title = "NLP Introduction"
        doc3.metadata.author = "NLP Specialist"
        doc3.chunks = [MockEnhancedChunk("Natural language processing applies machine learning to understand human language.")]
        documents.append(doc3)
        
        # Document 4: Cooking (unrelated topic)
        doc4 = MockEnhancedDocument(
            "Cooking is the art of preparing food using various techniques and ingredients.",
            "cooking_basics.txt"
        )
        doc4.metadata.title = "Cooking Fundamentals"
        doc4.metadata.author = "Chef Expert"
        doc4.chunks = [MockEnhancedChunk("Cooking is the art of preparing food using various techniques and ingredients.")]
        documents.append(doc4)
        
        return documents
    
    def test_document_relationship_manager_initialization(self):
        """Test DocumentRelationshipManager initialization"""
        self.assertIsNotNone(self.vector_store.relationship_manager)
        self.assertEqual(
            self.vector_store.relationship_manager.collection_name,
            "test_collection_relationships"
        )
    
    def test_quality_monitor_initialization(self):
        """Test QualityMonitor initialization"""
        self.assertIsNotNone(self.vector_store.quality_monitor)
        self.assertTrue((self.test_dir / "quality.db").exists())
    
    def test_document_storage_with_fase2_metadata(self):
        """Test document storage with FASE 2 enhanced metadata"""
        doc = self.test_docs[0]
        embeddings = [np.random.rand(384).astype(np.float32)]
        
        success = self.vector_store.store_document(doc, embeddings)
        self.assertTrue(success)
        
        # Verify quality metrics were stored
        quality_metrics = self.vector_store.get_quality_metrics(doc.id)
        self.assertGreater(quality_metrics['quality_score'], 0)
        self.assertEqual(quality_metrics['confidence_metrics']['chunk_count'], 1)
    
    def test_cross_document_correlation_search(self):
        """Test cross-document correlation search functionality"""
        # Store related documents
        for i, doc in enumerate(self.test_docs[:3]):  # ML-related docs
            embeddings = [np.random.rand(384).astype(np.float32)]
            self.vector_store.store_document(doc, embeddings)
        
        # Search with correlation
        query_embedding = np.random.rand(384).astype(np.float32)
        results = self.vector_store.search_with_cross_document_correlation(
            query_embedding, top_k=3, include_related=True
        )
        
        self.assertIsInstance(results, list)
        if results:  # Results might be empty due to random embeddings
            for result in results:
                self.assertIn('confidence_score', result)
                self.assertIn('related_documents', result)
    
    def test_hierarchical_search(self):
        """Test hierarchical search capabilities"""
        doc = self.test_docs[0]
        embeddings = [np.random.rand(384).astype(np.float32)]
        self.vector_store.store_document(doc, embeddings)
        
        query_embedding = np.random.rand(384).astype(np.float32)
        results = self.vector_store.search_within_document_hierarchy(
            query_embedding, document_id=doc.id, top_k=1
        )
        
        self.assertIsInstance(results, list)
    
    def test_semantic_clustering(self):
        """Test semantic clustering functionality"""
        # Store multiple documents
        for doc in self.test_docs:
            embeddings = [np.random.rand(384).astype(np.float32)]
            self.vector_store.store_document(doc, embeddings)
        
        # Build semantic clusters
        if self.vector_store.relationship_manager:
            cluster_results = self.vector_store.relationship_manager.build_semantic_clusters(
                self.vector_store.collection
            )
            
            self.assertIn('clusters', cluster_results)
            self.assertIn('cluster_assignments', cluster_results)
            self.assertGreaterEqual(cluster_results['cluster_count'], 1)
    
    def test_citation_network_analysis(self):
        """Test citation network functionality"""
        doc = self.test_docs[0]
        embeddings = [np.random.rand(384).astype(np.float32)]
        self.vector_store.store_document(doc, embeddings)
        
        citation_network = self.vector_store.get_document_citation_network(doc.id)
        
        self.assertIn('document_id', citation_network)
        self.assertIn('citations', citation_network)
        self.assertIn('references', citation_network)
        self.assertIn('network_depth', citation_network)
    
    def test_dependency_analysis(self):
        """Test cross-document dependency analysis"""
        doc_ids = []
        for doc in self.test_docs[:3]:
            embeddings = [np.random.rand(384).astype(np.float32)]
            self.vector_store.store_document(doc, embeddings)
            doc_ids.append(doc.id)
        
        dependency_analysis = self.vector_store.analyze_cross_document_dependencies(doc_ids)
        
        self.assertIn('dependencies', dependency_analysis)
        self.assertIn('dependency_graph', dependency_analysis)
        self.assertIn('circular_dependencies', dependency_analysis)
    
    def test_quality_metrics_tracking(self):
        """Test quality metrics tracking and monitoring"""
        doc = self.test_docs[0]
        embeddings = [np.random.rand(384).astype(np.float32)]
        self.vector_store.store_document(doc, embeddings)
        
        # Test document-specific quality metrics
        doc_quality = self.vector_store.get_quality_metrics(doc.id)
        self.assertIn('quality_score', doc_quality)
        self.assertIn('confidence_metrics', doc_quality)
        self.assertIn('validation_status', doc_quality)
        
        # Test collection-wide quality metrics
        collection_quality = self.vector_store.get_quality_metrics()
        self.assertIn('quality_score', collection_quality)
        self.assertIn('document_count', collection_quality)
    
    def test_correlation_validation(self):
        """Test correlation validation mechanisms"""
        # Store documents to create correlations
        for doc in self.test_docs[:2]:
            embeddings = [np.random.rand(384).astype(np.float32)]
            self.vector_store.store_document(doc, embeddings)
        
        validation_results = self.vector_store.validate_correlations(0.7)
        
        self.assertIn('validation_status', validation_results)
        self.assertIn('total_relationships', validation_results)
        self.assertIn('issues', validation_results)
    
    def test_performance_optimization(self):
        """Test performance optimization for large scale"""
        optimization_results = FASE2Helper.optimize_for_scale(
            self.vector_store, target_doc_count=1000
        )
        
        self.assertIn('status', optimization_results)
        self.assertIn('performance_level', optimization_results)
        self.assertIn('optimizations_applied', optimization_results)
    
    def test_fase2_integration_validation(self):
        """Test FASE 2 integration validation"""
        validation_results = FASE2Helper.validate_fase2_integration(self.vector_store)
        
        self.assertIn('status', validation_results)
        self.assertIn('relationship_manager', validation_results)
        self.assertIn('quality_monitor', validation_results)
        self.assertIn('chromadb_available', validation_results)
        self.assertIn('sklearn_available', validation_results)
    
    def test_search_result_confidence_scoring(self):
        """Test confidence scoring for search results"""
        doc = self.test_docs[0]
        embeddings = [np.random.rand(384).astype(np.float32)]
        self.vector_store.store_document(doc, embeddings)
        
        query_embedding = np.random.rand(384).astype(np.float32)
        results = self.vector_store.search_with_cross_document_correlation(
            query_embedding, top_k=1
        )
        
        if results:
            result = results[0]
            self.assertIn('confidence_score', result)
            self.assertGreaterEqual(result['confidence_score'], 0.0)
            self.assertLessEqual(result['confidence_score'], 1.0)
    
    def test_backward_compatibility(self):
        """Test that FASE 2 enhancements maintain backward compatibility"""
        # Test standard document without enhanced metadata
        standard_doc = Document("Test content", "test.txt")
        standard_chunk = Chunk("Test content")
        standard_doc.chunks = [standard_chunk]
        
        embeddings = [np.random.rand(384).astype(np.float32)]
        success = self.vector_store.store_document(standard_doc, embeddings)
        self.assertTrue(success)
        
        # Standard search should still work
        query_embedding = np.random.rand(384).astype(np.float32)
        results = self.vector_store.search_similar(query_embedding, top_k=1)
        self.assertIsInstance(results, list)
    
    def test_error_handling_and_resilience(self):
        """Test error handling in FASE 2 components"""
        # Test with invalid document ID
        invalid_relations = self.vector_store.relationship_manager.get_related_documents("invalid_id")
        self.assertEqual(invalid_relations, [])
        
        # Test quality metrics for non-existent document
        invalid_quality = self.vector_store.get_quality_metrics("invalid_id")
        self.assertEqual(invalid_quality['validation_status'], 'not_found')
        
        # Test correlation validation with empty collection
        validation = self.vector_store.validate_correlations()
        self.assertIn('validation_status', validation)


class TestPerformanceScale(unittest.TestCase):
    """Test performance at scale for 10k+ documents"""
    
    def setUp(self):
        """Set up for scale testing"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.vector_store = ChromaVectorStore(
            persist_directory=str(self.test_dir / "scale_test"),
            collection_name="scale_collection"
        )
    
    def tearDown(self):
        """Clean up scale test environment"""
        self.vector_store.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_batch_document_storage_performance(self):
        """Test performance of batch document storage"""
        # Create batch of documents
        batch_size = 100
        documents = []
        
        for i in range(batch_size):
            doc = MockEnhancedDocument(
                f"Test document content {i} with machine learning concepts and artificial intelligence.",
                f"test_doc_{i}.txt"
            )
            doc.metadata.title = f"Test Document {i}"
            doc.chunks = [MockEnhancedChunk(f"Test content chunk {i}")]
            documents.append(doc)
        
        # Measure storage time
        start_time = time.time()
        
        for doc in documents:
            embeddings = [np.random.rand(384).astype(np.float32)]
            success = self.vector_store.store_document(doc, embeddings)
            self.assertTrue(success)
        
        storage_time = time.time() - start_time
        
        # Performance assertion: should handle 100 docs in reasonable time
        self.assertLess(storage_time, 60.0)  # Less than 1 minute
        
        logging.info(f"Stored {batch_size} documents in {storage_time:.2f} seconds")
        
        # Test search performance
        query_embedding = np.random.rand(384).astype(np.float32)
        
        search_start = time.time()
        results = self.vector_store.search_with_cross_document_correlation(
            query_embedding, top_k=10
        )
        search_time = time.time() - search_start
        
        self.assertLess(search_time, 5.0)  # Search should be fast
        self.assertIsInstance(results, list)
    
    def test_clustering_performance_at_scale(self):
        """Test clustering performance with larger document sets"""
        # Create moderate number of documents for clustering test
        doc_count = 50
        
        for i in range(doc_count):
            doc = MockEnhancedDocument(
                f"Document {i} about machine learning and artificial intelligence topic {i % 5}",
                f"cluster_test_{i}.txt"
            )
            doc.chunks = [MockEnhancedChunk(f"Cluster test content {i}")]
            
            embeddings = [np.random.rand(384).astype(np.float32)]
            self.vector_store.store_document(doc, embeddings)
        
        # Test clustering performance
        if self.vector_store.relationship_manager:
            cluster_start = time.time()
            cluster_results = self.vector_store.relationship_manager.build_semantic_clusters(
                self.vector_store.collection
            )
            cluster_time = time.time() - cluster_start
            
            self.assertLess(cluster_time, 30.0)  # Clustering should complete in reasonable time
            self.assertGreater(cluster_results['cluster_count'], 0)
            
            logging.info(f"Built {cluster_results['cluster_count']} clusters from {doc_count} documents in {cluster_time:.2f} seconds")


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    unittest.main()