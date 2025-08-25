# Enhanced RAG Quality Framework

A comprehensive quality management system for enterprise RAG applications, providing unified quality assessment, monitoring, and optimization across document processing, vector storage, and search operations.

## Overview

The Enhanced RAG Quality Framework integrates seamlessly with DocumentProcessor v3 and VectorStore v3 to provide:

- **Comprehensive Quality Assessment**: Document processing, chunking, embedding, and search quality validation
- **Real-time Monitoring**: Continuous quality tracking with configurable alerts and thresholds
- **Cross-document Correlation**: Advanced relationship analysis and validation
- **Performance Optimization**: Automated quality-driven optimization recommendations
- **Enterprise Reporting**: Quality dashboards, audit reports, and compliance tracking
- **Scalability**: Designed for 10k+ document operations with minimal overhead

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Enhanced RAG Quality System                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │ Quality Manager │  │ Quality Monitors │  │   Validators    │ │
│  │                 │  │                  │  │                 │ │
│  │ • Coordination  │  │ • Real-time      │  │ • Content       │ │
│  │ • Assessment    │  │ • Alerting       │  │ • Chunking      │ │
│  │ • Reporting     │  │ • Trending       │  │ • Embeddings    │ │
│  └─────────────────┘  └──────────────────┘  │ • Search        │ │
│                                              └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    Component Integration                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │ DocumentProc v3 │  │  VectorStore v3  │  │  RAG Services   │ │
│  │                 │  │                  │  │                 │ │
│  │ • Processing    │  │ • Storage        │  │ • Query         │ │
│  │ • Extraction    │  │ • Search         │  │ • Response      │ │
│  │ • Chunking      │  │ • Correlation    │  │ • Generation    │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Quality Manager (`quality_framework.py`)
Central coordinator for quality assessment across all RAG components.

**Key Features:**
- Unified quality scoring system
- Component integration management
- Performance tracking and analysis
- Quality-driven optimization
- Enterprise audit reporting

### 2. Quality Validators (`quality_validators.py`)
Specialized validators for different aspects of the RAG pipeline.

**Validators:**
- `DocumentContentValidator`: Content quality and consistency
- `ChunkQualityValidator`: Chunking strategy and coverage
- `EmbeddingQualityValidator`: Vector representation quality
- `SearchQualityValidator`: Result relevance and diversity
- `ComprehensiveValidator`: Unified validation orchestration

### 3. Quality Monitoring (`quality_monitoring.py`)
Real-time monitoring service with alerting and trend analysis.

**Capabilities:**
- Configurable monitoring rules
- Real-time alert generation
- Performance trend analysis
- Email/Slack notifications
- Historical data analysis

### 4. Quality Integration (`quality_integration.py`)
Seamless integration layer with existing RAG components.

**Integration Features:**
- Automatic component registration
- Enhanced processing workflows
- Quality-aware search operations
- Dashboard data generation
- Custom monitoring configuration

## Quick Start

### 1. Basic Setup

```python
from core.quality_integration import EnhancedRAGQualitySystem
from core.quality_framework import ValidationLevel

# Initialize quality system
quality_system = EnhancedRAGQualitySystem(
    db_path="rag_quality.db",
    validation_level=ValidationLevel.COMPREHENSIVE,
    enable_monitoring=True,
    monitoring_interval=30  # seconds
)

# Register your RAG components
quality_system.register_document_processor(document_processor)
quality_system.register_vector_store(vector_store)
quality_system.register_embedding_service(embedding_service)

# Start monitoring
quality_system.start_monitoring()
```

### 2. Document Processing with Quality Assessment

```python
# Process document with comprehensive quality validation
result = quality_system.process_document_with_quality_assessment(
    file_path="document.pdf",
    filename="important_doc.pdf"
)

# Check quality results
print(f"Quality Score: {result['quality_score']:.3f}")
print(f"Quality Level: {result['quality_level']}")
print(f"Meets Standards: {result['meets_quality_standards']}")

# Review recommendations
for recommendation in result['quality_recommendations']:
    print(f"Recommendation: {recommendation}")
```

### 3. Search with Quality Assessment

```python
# Perform search with quality validation
search_result = quality_system.search_with_quality_assessment(
    query="How does quality monitoring work?",
    query_embedding=query_embedding,
    top_k=5
)

# Analyze search quality
print(f"Search Quality: {search_result['quality_score']:.3f}")
print(f"Results: {search_result['result_count']}")

# Enhanced results with quality metadata
for result in search_result['results']:
    print(f"Content: {result['content'][:100]}...")
    print(f"Confidence: {result['quality_metadata']['confidence_adjusted_similarity']:.3f}")
```

### 4. Pipeline Health Monitoring

```python
# Assess overall pipeline health
health_report = quality_system.assess_pipeline_health(include_trends=True)

print(f"System Status: {health_report['system_status']}")
print(f"Health Score: {health_report['overall_health_score']:.3f}")

# Check for issues
risk_assessment = health_report['risk_assessment']
if risk_assessment['high_risk_areas']:
    print("High Risk Areas:")
    for risk in risk_assessment['high_risk_areas']:
        print(f"  - {risk}")
```

## Advanced Configuration

### Custom Quality Thresholds

```python
quality_system = EnhancedRAGQualitySystem(
    quality_thresholds={
        'document_processing': 0.8,  # Higher standard
        'chunking_quality': 0.7,
        'embedding_quality': 0.8,
        'search_relevance': 0.7,
        'correlation_accuracy': 0.6,
        'overall_pipeline': 0.75
    }
)
```

### Email Alerting

```python
# Configure email alerts for critical issues
email_config = {
    'smtp_server': 'smtp.company.com',
    'smtp_port': 587,
    'username': 'rag-alerts@company.com',
    'password': 'secure-password',
    'recipients': ['ops-team@company.com', 'data-team@company.com']
}

quality_system.add_email_alerts(email_config)
```

### Custom Monitoring Rules

```python
from core.quality_monitoring import MonitoringRule, AlertSeverity

# Create custom rule for processing throughput
throughput_rule = MonitoringRule(
    rule_id="production_throughput_critical",
    component=ComponentType.DOCUMENT_PROCESSOR,
    metric_name="processing_throughput",
    threshold_value=10.0,  # docs/second
    comparison_operator="<",
    alert_severity=AlertSeverity.CRITICAL,
    consecutive_violations=3,
    evaluation_window=300  # 5 minutes
)

quality_system.add_custom_monitoring_rule(throughput_rule)
```

## Quality Metrics

### Document Processing Quality
- **Content Quality**: Text extraction accuracy, encoding issues, structure detection
- **Chunking Quality**: Semantic boundaries, coverage, consistency
- **Extraction Confidence**: OCR accuracy, metadata completeness, processing reliability

### Search Quality
- **Relevance Scoring**: Result-query alignment, similarity distribution
- **Result Diversity**: Unique documents, content variation
- **Cross-correlation**: Document relationship accuracy

### System Performance
- **Processing Throughput**: Documents processed per second
- **Search Latency**: Average query response time
- **Error Rates**: Processing failures, search failures
- **Resource Usage**: Memory consumption, storage efficiency

## Quality Levels

| Level | Score Range | Description |
|-------|-------------|-------------|
| **Excellent** | ≥ 0.9 | Production-ready, enterprise quality |
| **Good** | 0.7 - 0.89 | Acceptable for most use cases |
| **Acceptable** | 0.5 - 0.69 | Minimum viable quality |
| **Poor** | 0.3 - 0.49 | Requires improvement |
| **Critical** | < 0.3 | System intervention needed |

## Dashboard and Reporting

### Quality Dashboard

```python
# Generate dashboard data
dashboard_data = quality_system.generate_quality_dashboard_data()

# Key dashboard components:
# - Real-time quality metrics
# - Active alerts and trends
# - Component health status
# - Performance summaries
# - Quality recommendations
```

### Audit Reports

```python
# Generate compliance audit report
audit_report = quality_system.quality_manager.generate_quality_audit_report(
    audit_period_days=30,
    include_compliance=True
)

# Report sections:
# - Executive summary
# - Quality metrics analysis
# - Performance analysis
# - Compliance status
# - Incidents and resolutions
# - Strategic recommendations
```

## Integration with Existing Components

### DocumentProcessor v3 Integration

The quality framework automatically enhances DocumentProcessor v3 with:
- Quality assessment of extracted content
- Confidence scoring for processing operations
- Validation of chunking strategies
- Performance monitoring

### VectorStore v3 Integration

Enhanced VectorStore v3 capabilities:
- Search result quality assessment
- Cross-document correlation validation
- Embedding quality monitoring
- Performance tracking

### Custom Component Integration

```python
# Register custom RAG components
quality_system.register_component(
    component_type=ComponentType.CUSTOM,
    component_instance=your_component
)
```

## Production Deployment

### System Requirements
- **Memory**: 512MB minimum for quality framework overhead
- **Storage**: 1GB for quality databases and logs (per 10k documents)
- **CPU**: Minimal impact (<5% overhead in typical deployments)

### Scaling Considerations
- **Document Volume**: Tested with 10k+ documents
- **Monitoring Frequency**: Configurable (recommended: 30-60 seconds)
- **Data Retention**: Configurable (default: 90 days for alerts, 30 days for metrics)

### Best Practices

1. **Quality Thresholds**: Start with default thresholds and adjust based on your data
2. **Monitoring Rules**: Begin with standard rules, add custom rules for specific needs
3. **Alert Configuration**: Set up email alerts for critical issues only
4. **Regular Audits**: Generate monthly quality audit reports
5. **Performance Monitoring**: Monitor system impact and adjust monitoring frequency

## Troubleshooting

### Common Issues

**Q: Quality scores are consistently low**
A: Check quality thresholds, review document preprocessing, validate embedding model selection

**Q: Monitoring service not starting**
A: Verify database permissions, check port availability, review configuration settings

**Q: High memory usage**
A: Adjust monitoring frequency, reduce data retention periods, optimize database queries

**Q: Integration failures**
A: Verify component compatibility, check import paths, review component registration

### Debug Logging

```python
import logging

# Enable debug logging for quality framework
logging.getLogger('core.quality_framework').setLevel(logging.DEBUG)
logging.getLogger('core.quality_monitoring').setLevel(logging.DEBUG)
```

## API Reference

### Core Classes

- **`EnhancedRAGQualitySystem`**: Main quality system coordinator
- **`QualityManager`**: Central quality assessment and management
- **`QualityMonitoringService`**: Real-time monitoring and alerting
- **`ComprehensiveValidator`**: Unified validation orchestration

### Key Methods

```python
# Document processing
process_document_with_quality_assessment(file_path, filename, embeddings)

# Search operations
search_with_quality_assessment(query, query_embedding, top_k, expected_results)

# System health
assess_pipeline_health(include_trends)

# Quality optimization
optimize_quality_settings(target_quality, focus_areas)

# Dashboard data
generate_quality_dashboard_data()
```

## Examples

See [`examples/quality_framework_example.py`](../examples/quality_framework_example.py) for a comprehensive demonstration of all quality framework capabilities.

## Support and Contributing

For questions, issues, or contributions:
1. Check the example script for usage patterns
2. Review the API documentation
3. Submit issues with detailed error logs
4. Include system configuration in bug reports

## License

This quality framework is part of the Enhanced RAG System and follows the same licensing terms as the parent project.