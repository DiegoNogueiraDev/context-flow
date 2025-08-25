#!/usr/bin/env python3
"""
Quick integration test for the Enhanced RAG Quality Framework

This script verifies that the quality framework integrates properly with
existing DocumentProcessor v3 and VectorStore v3 components.
"""

import sys
import os
from pathlib import Path
import tempfile
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_quality_framework_imports():
    """Test that all quality framework components import correctly"""
    print("Testing Quality Framework imports...")
    
    try:
        from core.quality_framework import QualityManager, ValidationLevel, QualityLevel
        print("✓ Quality Framework core imports successful")
        
        from core.quality_validators import ComprehensiveValidator, ValidationSeverity
        print("✓ Quality Validators imports successful")
        
        from core.quality_monitoring import QualityMonitoringService, AlertSeverity
        print("✓ Quality Monitoring imports successful")
        
        from core.quality_integration import EnhancedRAGQualitySystem
        print("✓ Quality Integration imports successful")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_quality_manager_initialization():
    """Test Quality Manager initialization"""
    print("\nTesting Quality Manager initialization...")
    
    try:
        from core.quality_framework import QualityManager, ValidationLevel
        
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_quality.db")
            
            quality_manager = QualityManager(
                db_path=db_path,
                validation_level=ValidationLevel.STANDARD,
                enable_real_time_monitoring=False  # Disable for testing
            )
            
            print("✓ Quality Manager initialized successfully")
            
            # Test basic functionality
            pipeline_report = quality_manager.assess_pipeline_quality(include_trends=False)
            print("✓ Pipeline quality assessment working")
            
            # Clean up
            quality_manager.close()
            print("✓ Quality Manager cleanup successful")
            
        return True
    except Exception as e:
        print(f"✗ Quality Manager test failed: {e}")
        return False

def test_document_processor_integration():
    """Test integration with DocumentProcessor v3"""
    print("\nTesting DocumentProcessor v3 integration...")
    
    try:
        from core.document_processor import DocumentProcessor
        from core.quality_integration import EnhancedRAGQualitySystem
        from core.quality_framework import ValidationLevel
        from core.models import Document, Chunk
        
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_integration.db")
            
            # Initialize quality system
            quality_system = EnhancedRAGQualitySystem(
                db_path=db_path,
                validation_level=ValidationLevel.STANDARD,
                enable_monitoring=False  # Disable for testing
            )
            
            # Initialize document processor
            doc_processor = DocumentProcessor(
                use_docling=False,  # Use fallback for testing
                enable_ocr=False,
                enable_table_extraction=False,
                enable_figure_extraction=False
            )
            
            # Register with quality system
            enhanced_processor = quality_system.register_document_processor(doc_processor)
            print("✓ DocumentProcessor registered with quality system")
            
            # Create test document
            test_content = "This is a test document for quality framework integration testing."
            test_doc = Document(test_content, "test_doc.txt")
            test_doc.chunks = [Chunk(test_content, 0, len(test_content))]
            
            # Assess document quality
            quality_report = quality_system.quality_manager.assess_document_quality(test_doc)
            print(f"✓ Document quality assessment completed (score: {quality_report.overall_score:.3f})")
            
            # Clean up
            quality_system.close()
            
        return True
    except Exception as e:
        print(f"✗ DocumentProcessor integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_store_integration():
    """Test integration with VectorStore components"""
    print("\nTesting VectorStore integration...")
    
    try:
        # Try to import vector stores
        vector_store = None
        vector_store_type = "none"
        
        try:
            from storage.vector_store_v2 import ChromaVectorStore
            with tempfile.TemporaryDirectory() as temp_dir:
                vector_store = ChromaVectorStore(persist_directory=temp_dir, collection_name="test")
                vector_store_type = "ChromaVectorStore"
        except ImportError:
            try:
                from storage.vector_store import VectorStore
                with tempfile.TemporaryDirectory() as temp_dir:
                    vector_store = VectorStore(os.path.join(temp_dir, "test.db"))
                    vector_store.initialize_database()
                    vector_store_type = "SQLiteVectorStore"
            except ImportError:
                print("⚠ No vector store available for testing")
                return True  # Skip test if no vector store
        
        if vector_store:
            from core.quality_integration import EnhancedRAGQualitySystem
            from core.quality_framework import ValidationLevel
            
            with tempfile.TemporaryDirectory() as temp_dir:
                db_path = os.path.join(temp_dir, "test_vector_integration.db")
                
                # Initialize quality system
                quality_system = EnhancedRAGQualitySystem(
                    db_path=db_path,
                    validation_level=ValidationLevel.STANDARD,
                    enable_monitoring=False
                )
                
                # Register vector store
                enhanced_vector_store = quality_system.register_vector_store(vector_store)
                print(f"✓ {vector_store_type} registered with quality system")
                
                # Test quality assessment
                import numpy as np
                query_embedding = np.random.normal(0, 1, 100)
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
                
                # Mock search results for testing
                mock_results = [
                    {'chunk_id': 'test1', 'content': 'Test content 1', 'similarity': 0.8},
                    {'chunk_id': 'test2', 'content': 'Test content 2', 'similarity': 0.6}
                ]
                
                # Test search quality assessment
                search_report = quality_system.quality_manager.assess_search_quality(
                    query="test query",
                    results=mock_results,
                    query_embedding=query_embedding
                )
                
                print(f"✓ Search quality assessment completed (score: {search_report.overall_score:.3f})")
                
                # Clean up
                quality_system.close()
                if hasattr(vector_store, 'close'):
                    vector_store.close()
        
        return True
    except Exception as e:
        print(f"✗ VectorStore integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quality_validators():
    """Test quality validators"""
    print("\nTesting Quality Validators...")
    
    try:
        from core.quality_validators import (
            DocumentContentValidator, 
            ChunkQualityValidator, 
            ComprehensiveValidator
        )
        from core.models import Document, Chunk
        
        # Create test document
        test_content = "This is a comprehensive test document for quality validation. " * 10
        test_doc = Document(test_content, "validator_test.txt")
        test_doc.chunks = [
            Chunk(test_content[:100], 0, 100),
            Chunk(test_content[100:200], 100, 200),
            Chunk(test_content[200:], 200, len(test_content))
        ]
        
        # Test content validator
        content_validator = DocumentContentValidator()
        content_result = content_validator.validate(test_doc)
        print(f"✓ Content validation completed (score: {content_result.overall_score:.3f})")
        
        # Test chunk validator
        chunk_validator = ChunkQualityValidator()
        chunk_result = chunk_validator.validate(test_doc)
        print(f"✓ Chunk validation completed (score: {chunk_result.overall_score:.3f})")
        
        # Test comprehensive validator
        comprehensive_validator = ComprehensiveValidator()
        validation_results = comprehensive_validator.validate_document_processing(test_doc)
        
        report = comprehensive_validator.generate_comprehensive_report(validation_results)
        print(f"✓ Comprehensive validation completed (overall score: {report['overall_score']:.3f})")
        
        return True
    except Exception as e:
        print(f"✗ Quality validators test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_monitoring_service():
    """Test quality monitoring service"""
    print("\nTesting Quality Monitoring Service...")
    
    try:
        from core.quality_monitoring import QualityMonitoringService
        from core.quality_framework import QualityMetric, ComponentType
        
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_monitoring.db")
            
            # Initialize monitoring service (but don't start monitoring loop)
            monitoring_service = QualityMonitoringService(
                db_path=db_path,
                monitoring_interval=60  # Long interval for testing
            )
            
            print("✓ Monitoring service initialized successfully")
            
            # Test metric recording
            test_metric = QualityMetric(
                name="test_metric",
                value=0.85,
                confidence=0.9,
                component=ComponentType.DOCUMENT_PROCESSOR
            )
            
            monitoring_service.record_quality_metric(test_metric)
            print("✓ Quality metric recorded successfully")
            
            # Test system health summary
            health_summary = monitoring_service.get_system_health_summary()
            print(f"✓ System health summary generated (status: {health_summary['system_status']})")
            
            # Clean up
            monitoring_service.close()
            
        return True
    except Exception as e:
        print(f"✗ Monitoring service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests"""
    print("Enhanced RAG Quality Framework Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_quality_framework_imports),
        ("Quality Manager", test_quality_manager_initialization),
        ("DocumentProcessor Integration", test_document_processor_integration),
        ("VectorStore Integration", test_vector_store_integration),
        ("Quality Validators", test_quality_validators),
        ("Monitoring Service", test_monitoring_service)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:.<40} {status}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print("-" * 60)
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Quality Framework is ready for use.")
        return 0
    else:
        print(f"\n⚠ {failed} test(s) failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)