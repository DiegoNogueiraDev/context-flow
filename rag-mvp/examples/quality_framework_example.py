#!/usr/bin/env python3
"""
Comprehensive Quality Framework Example for Enhanced RAG System

This example demonstrates how to use the Quality Framework with DocumentProcessor v3
and VectorStore v3 for enterprise-grade quality assurance and monitoring.
"""

import sys
import os
import numpy as np
import time
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.quality_integration import EnhancedRAGQualitySystem
from core.quality_framework import ValidationLevel, QualityLevel, ComponentType
from core.quality_monitoring import MonitoringRule, AlertSeverity
from core.document_processor import DocumentProcessor
from core.models import Document, Chunk

# Import vector store (adjust import based on your setup)
try:
    from storage.vector_store_v2 import ChromaVectorStore as VectorStore
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    try:
        from storage.vector_store import VectorStore
        VECTOR_STORE_AVAILABLE = True
    except ImportError:
        VECTOR_STORE_AVAILABLE = False
        print("Warning: Vector store not available for example")


def create_sample_document() -> Document:
    """Create a sample document for testing"""
    content = """
# Sample Document for Quality Testing

## Introduction

This is a comprehensive sample document designed to test the quality framework
of our enhanced RAG system. It contains various types of content including
headers, paragraphs, and structured information.

## Main Content

### Section A: Technical Overview

The quality framework provides real-time monitoring and assessment of document
processing, embedding generation, and search operations. It includes:

- Document processing quality validation
- Embedding consistency checking
- Search result relevance scoring
- Cross-document correlation analysis

### Section B: Performance Metrics

Performance is measured across multiple dimensions:

1. Processing throughput (documents per second)
2. Search latency (response time)
3. Memory usage (system resources)
4. Error rates (failure percentage)
5. Availability (uptime percentage)

### Section C: Quality Standards

The system maintains quality standards through:

* Automated validation checks
* Real-time monitoring alerts
* Trend analysis and forecasting
* Continuous optimization recommendations

## Conclusion

This quality framework ensures enterprise-grade reliability and performance
for production RAG systems at scale.
"""
    
    document = Document(content, "sample_quality_test.md")
    
    # Create sample chunks
    chunks = [
        Chunk("# Sample Document for Quality Testing\n\n## Introduction", 0, 57),
        Chunk("This is a comprehensive sample document designed to test the quality framework", 59, 139),
        Chunk("of our enhanced RAG system. It contains various types of content including", 140, 215),
        Chunk("headers, paragraphs, and structured information.", 216, 265),
        Chunk("## Main Content\n\n### Section A: Technical Overview", 267, 315),
        Chunk("The quality framework provides real-time monitoring and assessment", 317, 383),
        Chunk("of document processing, embedding generation, and search operations.", 384, 453),
        Chunk("It includes:\n\n- Document processing quality validation", 454, 502),
        Chunk("- Embedding consistency checking", 503, 536),
        Chunk("- Search result relevance scoring", 537, 569),
        Chunk("- Cross-document correlation analysis", 570, 607)
    ]
    
    document.chunks = chunks
    return document


def create_sample_embeddings(num_chunks: int) -> list:
    """Create sample embeddings for testing"""
    # Generate random embeddings (384 dimensions - typical for sentence transformers)
    np.random.seed(42)  # For reproducible results
    embeddings = []
    
    for i in range(num_chunks):
        # Create somewhat realistic embeddings with variation
        base_embedding = np.random.normal(0, 0.1, 384)
        # Add some structure to make embeddings somewhat similar but distinct
        base_embedding[i*10:(i*10+10)] += np.random.normal(0, 0.5, 10)
        # Normalize
        base_embedding = base_embedding / np.linalg.norm(base_embedding)
        embeddings.append(base_embedding)
    
    return embeddings


def demonstrate_document_quality_assessment():
    """Demonstrate comprehensive document quality assessment"""
    print("=" * 80)
    print("DOCUMENT QUALITY ASSESSMENT DEMONSTRATION")
    print("=" * 80)
    
    # Initialize quality system
    quality_system = EnhancedRAGQualitySystem(
        db_path="example_quality_demo.db",
        validation_level=ValidationLevel.COMPREHENSIVE,
        enable_monitoring=True,
        monitoring_interval=30
    )
    
    # Initialize document processor
    print("\n1. Initializing DocumentProcessor v3...")
    document_processor = DocumentProcessor(
        use_docling=False,  # Use fallback for demo
        enable_ocr=False,
        enable_table_extraction=True,
        enable_figure_extraction=True
    )
    
    # Register with quality system
    enhanced_processor = quality_system.register_document_processor(document_processor)
    
    # Create sample document
    print("2. Creating sample document...")
    document = create_sample_document()
    
    # Generate sample embeddings
    print("3. Generating sample embeddings...")
    embeddings = create_sample_embeddings(len(document.chunks))
    
    # Process document with quality assessment
    print("4. Processing document with comprehensive quality assessment...")
    start_time = time.time()
    
    result = quality_system.process_document_with_quality_assessment(
        file_path="",  # Not used in this demo
        filename="sample_quality_test.md",
        embeddings=embeddings
    )
    
    processing_time = time.time() - start_time
    
    # Display results
    print(f"\n5. QUALITY ASSESSMENT RESULTS (processed in {processing_time:.2f}s)")
    print("-" * 60)
    
    print(f"Overall Quality Score: {result['quality_score']:.3f}")
    print(f"Quality Level: {result['quality_level']}")
    print(f"Meets Standards: {result['meets_quality_standards']}")
    print(f"Processing Time: {result['processing_time']:.3f}s")
    
    # Component scores
    print("\nComponent Quality Scores:")
    for component, score in result['quality_report'].component_scores.items():
        print(f"  {component.value}: {score:.3f}")
    
    # Confidence scores
    print("\nConfidence Scores:")
    for metric, score in result['quality_report'].confidence_scores.items():
        print(f"  {metric}: {score:.3f}")
    
    # Validation summary
    validation_summary = result['validation_summary']
    print(f"\nValidation Summary:")
    print(f"  Components Validated: {validation_summary['validation_summary']['components_validated']}")
    print(f"  Components Passed: {validation_summary['validation_summary']['components_passed']}")
    print(f"  Critical Issues: {validation_summary['issues_by_severity']['critical']}")
    print(f"  Total Issues: {validation_summary['total_issues']}")
    
    # Alerts and recommendations
    if result['quality_report'].alerts:
        print("\nQuality Alerts:")
        for alert in result['quality_report'].alerts[:3]:  # Show first 3
            print(f"  - {alert}")
    
    if result['quality_recommendations']:
        print("\nRecommendations:")
        for rec in result['quality_recommendations'][:3]:  # Show first 3
            print(f"  - {rec}")
    
    return quality_system, result


def demonstrate_search_quality_assessment(quality_system):
    """Demonstrate search quality assessment"""
    print("\n" + "=" * 80)
    print("SEARCH QUALITY ASSESSMENT DEMONSTRATION")
    print("=" * 80)
    
    if not VECTOR_STORE_AVAILABLE:
        print("Vector store not available - skipping search demo")
        return
    
    # Initialize vector store
    print("1. Initializing VectorStore v3...")
    vector_store = VectorStore(
        persist_directory="example_chroma_demo",
        collection_name="quality_demo"
    )
    
    # Register with quality system
    enhanced_vector_store = quality_system.register_vector_store(vector_store)
    
    # Create sample query and embedding
    print("2. Creating sample search query...")
    query = "What is the quality framework and how does it work?"
    query_embedding = np.random.normal(0, 0.1, 384)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Create sample search results
    print("3. Simulating search results...")
    sample_results = [
        {
            'chunk_id': 'chunk_1',
            'document_id': 'doc_1',
            'content': 'The quality framework provides real-time monitoring and assessment of document processing',
            'filename': 'sample_doc.md',
            'similarity': 0.85
        },
        {
            'chunk_id': 'chunk_2', 
            'document_id': 'doc_1',
            'content': 'It includes document processing quality validation and embedding consistency checking',
            'filename': 'sample_doc.md',
            'similarity': 0.78
        },
        {
            'chunk_id': 'chunk_3',
            'document_id': 'doc_2', 
            'content': 'Performance is measured across multiple dimensions including throughput and latency',
            'filename': 'performance_doc.md',
            'similarity': 0.65
        }
    ]
    
    # Perform search with quality assessment
    print("4. Performing search with quality assessment...")
    
    # Mock the vector store search method for demo
    original_search = getattr(vector_store, 'search_similar', None)
    vector_store.search_similar = lambda *args, **kwargs: sample_results
    
    try:
        search_result = quality_system.search_with_quality_assessment(
            query=query,
            query_embedding=query_embedding,
            top_k=3,
            expected_results=['doc_1', 'doc_2']  # For precision/recall calculation
        )
        
        # Display search quality results
        print("\n5. SEARCH QUALITY ASSESSMENT RESULTS")
        print("-" * 60)
        
        print(f"Query: {search_result['query']}")
        print(f"Results Returned: {search_result['result_count']}")
        print(f"Search Time: {search_result['search_time']:.3f}s")
        print(f"Quality Score: {search_result['quality_score']:.3f}")
        print(f"Quality Level: {search_result['quality_level']}")
        
        # Search-specific metrics
        quality_report = search_result['quality_report']
        print("\nSearch Quality Metrics:")
        for metric, score in quality_report.confidence_scores.items():
            print(f"  {metric}: {score:.3f}")
        
        # Enhanced results with quality metadata
        print("\nEnhanced Search Results:")
        for i, result in enumerate(search_result['results'][:2]):  # Show first 2
            print(f"  Result {i+1}:")
            print(f"    Content: {result['content'][:80]}...")
            print(f"    Similarity: {result['similarity']:.3f}")
            print(f"    Quality-Adjusted: {result['quality_metadata']['confidence_adjusted_similarity']:.3f}")
        
        if search_result['search_recommendations']:
            print("\nSearch Recommendations:")
            for rec in search_result['search_recommendations'][:2]:
                print(f"  - {rec}")
    
    finally:
        # Restore original search method
        if original_search:
            vector_store.search_similar = original_search
    
    print("\n6. Cleaning up vector store...")
    try:
        vector_store.close()
    except:
        pass


def demonstrate_pipeline_health_assessment(quality_system):
    """Demonstrate comprehensive pipeline health assessment"""
    print("\n" + "=" * 80)
    print("PIPELINE HEALTH ASSESSMENT DEMONSTRATION")
    print("=" * 80)
    
    print("1. Assessing overall pipeline health...")
    
    health_report = quality_system.assess_pipeline_health(include_trends=True)
    
    print("\n2. PIPELINE HEALTH REPORT")
    print("-" * 60)
    
    print(f"Overall Health Score: {health_report['overall_health_score']:.3f}")
    print(f"System Status: {health_report['system_status']}")
    
    # Component health
    monitoring_health = health_report.get('monitoring_health', {})
    component_health = monitoring_health.get('component_health', {})
    
    print("\nComponent Health:")
    for component, health in component_health.items():
        status = health.get('status', 'unknown')
        score = health.get('score', 0)
        print(f"  {component}: {status} (score: {score:.3f})")
    
    # Active alerts
    alert_summary = monitoring_health.get('alert_summary', {})
    print(f"\nActive Alerts:")
    print(f"  Critical: {alert_summary.get('critical', 0)}")
    print(f"  High: {alert_summary.get('high', 0)}")
    print(f"  Medium: {alert_summary.get('medium', 0)}")
    print(f"  Low: {alert_summary.get('low', 0)}")
    
    # Risk assessment
    risk_assessment = health_report.get('risk_assessment', {})
    print(f"\nRisk Assessment:")
    print(f"  Risk Score: {risk_assessment.get('risk_score', 0):.3f}")
    print(f"  High Risk Areas: {len(risk_assessment.get('high_risk_areas', []))}")
    print(f"  Medium Risk Areas: {len(risk_assessment.get('medium_risk_areas', []))}")
    
    # Strategic recommendations
    strategic_recs = health_report.get('strategic_recommendations', [])
    if strategic_recs:
        print("\nStrategic Recommendations:")
        for rec in strategic_recs[:3]:  # Show first 3
            print(f"  - {rec}")


def demonstrate_quality_optimization(quality_system):
    """Demonstrate quality optimization capabilities"""
    print("\n" + "=" * 80)
    print("QUALITY OPTIMIZATION DEMONSTRATION")
    print("=" * 80)
    
    print("1. Optimizing quality settings for GOOD quality level...")
    
    optimization_result = quality_system.optimize_quality_settings(
        target_quality=QualityLevel.GOOD,
        focus_areas=['document_processing', 'vector_store']
    )
    
    print("\n2. OPTIMIZATION RESULTS")
    print("-" * 60)
    
    print(f"Target Quality: {optimization_result['target_quality']}")
    print(f"Current Score: {optimization_result['current_score']:.3f}")
    print(f"Estimated Improvement: {optimization_result['estimated_improvement']:.3f}")
    
    # Component optimizations
    component_opts = optimization_result.get('component_optimizations', {})
    if component_opts:
        print("\nComponent Optimizations:")
        for component, opts in component_opts.items():
            print(f"  {component}:")
            for opt in opts[:2]:  # Show first 2
                print(f"    - {opt}")
    
    # Monitoring adjustments
    monitoring_adjustments = optimization_result.get('monitoring_adjustments', {})
    new_rules = monitoring_adjustments.get('new_rules', [])
    if new_rules:
        print(f"\nNew Monitoring Rules Added: {len(new_rules)}")
        for rule in new_rules:
            print(f"  - {rule}")


def demonstrate_quality_dashboard(quality_system):
    """Demonstrate quality dashboard data generation"""
    print("\n" + "=" * 80)
    print("QUALITY DASHBOARD DEMONSTRATION")
    print("=" * 80)
    
    print("1. Generating comprehensive dashboard data...")
    
    dashboard_data = quality_system.generate_quality_dashboard_data()
    
    print("\n2. DASHBOARD DATA SUMMARY")
    print("-" * 60)
    
    # System metadata
    metadata = dashboard_data.get('system_metadata', {})
    print(f"Validation Level: {metadata.get('validation_level', 'unknown')}")
    print(f"Monitoring Enabled: {metadata.get('monitoring_enabled', False)}")
    print(f"Components Registered: {metadata.get('components_registered', 0)}")
    
    # Quality overview
    quality_overview = dashboard_data.get('quality_overview', {})
    print(f"System Status: {quality_overview.get('system_status', 'unknown')}")
    
    # Current metrics
    current_metrics = quality_overview.get('current_metrics', {})
    if current_metrics:
        print("\nCurrent Quality Metrics:")
        for metric, value in list(current_metrics.items())[:3]:  # Show first 3
            print(f"  {metric}: {value}")
    
    # Performance summary
    performance_summary = dashboard_data.get('performance_summary', {})
    current_perf = performance_summary.get('current_performance', {})
    if current_perf:
        print("\nPerformance Summary:")
        print(f"  Processing Throughput: {current_perf.get('average_processing_time', 'N/A')}")
        print(f"  Search Latency: {current_perf.get('average_search_latency', 'N/A')}")
        print(f"  System Availability: {current_perf.get('system_availability', 'N/A')}")
        print(f"  Error Rate: {current_perf.get('error_rate', 'N/A')}")
    
    print(f"\nDashboard Generated At: {dashboard_data.get('dashboard_generated_at', 'unknown')}")


def demonstrate_custom_monitoring_rules(quality_system):
    """Demonstrate custom monitoring rules"""
    print("\n" + "=" * 80)
    print("CUSTOM MONITORING RULES DEMONSTRATION")
    print("=" * 80)
    
    print("1. Adding custom monitoring rules...")
    
    # Custom rule for embedding quality
    embedding_rule = MonitoringRule(
        rule_id="custom_embedding_quality_strict",
        component=ComponentType.EMBEDDING_SERVICE,
        metric_name="embedding_consistency",
        threshold_value=0.85,
        comparison_operator="<",
        alert_severity=AlertSeverity.HIGH,
        consecutive_violations=2,
        evaluation_window=300,  # 5 minutes
        metadata={
            'description': 'Strict embedding quality monitoring for production',
            'business_impact': 'high',
            'auto_resolve': False
        }
    )
    
    quality_system.add_custom_monitoring_rule(embedding_rule)
    
    # Custom rule for processing throughput
    throughput_rule = MonitoringRule(
        rule_id="custom_processing_throughput_low",
        component=ComponentType.DOCUMENT_PROCESSOR,
        metric_name="processing_throughput",
        threshold_value=5.0,  # documents per second
        comparison_operator="<",
        alert_severity=AlertSeverity.MEDIUM,
        consecutive_violations=3,
        evaluation_window=180,  # 3 minutes
        metadata={
            'description': 'Monitor processing throughput for performance issues',
            'business_impact': 'medium'
        }
    )
    
    quality_system.add_custom_monitoring_rule(throughput_rule)
    
    print("2. Custom monitoring rules added:")
    print(f"  - {embedding_rule.rule_id}")
    print(f"  - {throughput_rule.rule_id}")
    
    print("\n3. Starting quality monitoring...")
    quality_system.start_monitoring()
    
    print("4. Monitoring service is now active and will evaluate rules every 30 seconds")
    print("   In a production environment, this would continuously monitor quality metrics")


def main():
    """Main demonstration function"""
    print("ENHANCED RAG QUALITY FRAMEWORK COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    print("This demonstration shows the complete Quality Framework integration")
    print("with DocumentProcessor v3 and VectorStore v3 for enterprise RAG systems.")
    print()
    
    try:
        # 1. Document Quality Assessment
        quality_system, doc_result = demonstrate_document_quality_assessment()
        
        # 2. Search Quality Assessment  
        demonstrate_search_quality_assessment(quality_system)
        
        # 3. Pipeline Health Assessment
        demonstrate_pipeline_health_assessment(quality_system)
        
        # 4. Quality Optimization
        demonstrate_quality_optimization(quality_system)
        
        # 5. Dashboard Data Generation
        demonstrate_quality_dashboard(quality_system)
        
        # 6. Custom Monitoring Rules
        demonstrate_custom_monitoring_rules(quality_system)
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print()
        print("The Enhanced RAG Quality Framework provides:")
        print("✓ Comprehensive document processing quality assessment")
        print("✓ Real-time search result quality validation")
        print("✓ Pipeline-wide health monitoring and alerting")
        print("✓ Automated quality optimization recommendations")
        print("✓ Enterprise-grade quality reporting and dashboards")
        print("✓ Configurable monitoring rules and thresholds")
        print("✓ Integration with existing RAG system components")
        print()
        print("For production deployment:")
        print("1. Configure appropriate quality thresholds for your use case")
        print("2. Set up email/Slack alerting for critical quality issues")
        print("3. Implement regular quality audits and reporting")
        print("4. Monitor quality trends and optimize based on metrics")
        print("5. Scale monitoring rules based on document volume and criticality")
        
        # Keep monitoring running for a short time to show it's working
        print(f"\n6. Monitoring will continue for 60 seconds to demonstrate real-time capabilities...")
        time.sleep(60)
        
        # Clean up
        print("\n7. Cleaning up resources...")
        quality_system.close()
        
        print("\nDemonstration completed. Check the generated database files for stored quality data.")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)