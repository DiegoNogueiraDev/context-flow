"""
Tests for FASE 2 advanced DocumentProcessor capabilities.

This test file demonstrates the enhanced features while ensuring
backward compatibility is maintained.
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.document_processor import (
    DocumentProcessor, 
    EnhancedDocument, 
    DocumentMetadata,
    EnhancedChunk,
    DocumentQualityAssessor,
    SemanticChunker,
    ProcessingQuality,
    ChunkType,
    ExtractionConfidence,
    ProcessingStats,
    TableSchema,
    DocumentHierarchy
)


class TestFASE2DocumentProcessor:
    """Test FASE 2 advanced capabilities"""
    
    @pytest.fixture
    def advanced_processor(self):
        """Create DocumentProcessor with advanced features enabled"""
        return DocumentProcessor(
            use_docling=False,  # Use fallback for testing
            enable_ocr=True,
            enable_table_extraction=True,
            enable_figure_extraction=True
        )
    
    @pytest.fixture
    def quality_assessor(self):
        """Create DocumentQualityAssessor for testing"""
        return DocumentQualityAssessor()
    
    @pytest.fixture
    def semantic_chunker(self):
        """Create SemanticChunker for testing"""
        return SemanticChunker()
    
    @pytest.fixture
    def complex_markdown_content(self):
        """Complex markdown content for testing advanced features"""
        return """---
title: Advanced Machine Learning Research
author: Dr. Jane Smith
date: 2024-01-15
---

# Introduction

This document explores advanced machine learning techniques and their applications
in real-world scenarios.

## Background and Methodology

### Data Collection
Our methodology involves collecting data from multiple sources:
- Web scraping techniques
- API integrations  
- Manual data entry

### Data Processing
The data processing pipeline includes:

```python
def process_data(raw_data):
    cleaned_data = clean(raw_data)
    return transform(cleaned_data)
```

## Results and Analysis

### Performance Metrics

| Algorithm | Accuracy | Precision | Recall |
|-----------|----------|-----------|--------|
| Random Forest | 0.87 | 0.84 | 0.89 |
| SVM | 0.82 | 0.79 | 0.85 |
| Neural Network | 0.91 | 0.88 | 0.93 |

The results show that neural networks perform best overall.

### Discussion

The findings have several implications:
1. Neural networks are most effective for this task
2. Feature engineering is crucial
3. Cross-validation prevents overfitting

> Note: These results are preliminary and require further validation.

## Conclusions

This research demonstrates the effectiveness of modern machine learning
approaches for complex data analysis tasks.

### Future Work

Future research directions include:
- Exploring deep learning architectures
- Implementing real-time processing
- Scaling to larger datasets
"""
    
    def test_advanced_processor_initialization(self, advanced_processor):
        """Test that advanced processor initializes with all components"""
        assert advanced_processor is not None
        assert hasattr(advanced_processor, 'quality_assessor')
        assert hasattr(advanced_processor, 'semantic_chunker')
        assert advanced_processor.enable_ocr
        assert advanced_processor.enable_table_extraction
        assert advanced_processor.enable_figure_extraction
        assert advanced_processor.max_retries == 3
        assert advanced_processor.processing_timeout == 300
    
    def test_enhanced_document_creation(self, complex_markdown_content):
        """Test creation of EnhancedDocument with advanced metadata"""
        doc = EnhancedDocument(complex_markdown_content, "test.md")
        
        assert isinstance(doc.metadata, DocumentMetadata)
        assert doc.metadata.processing_version == "2.0"
        assert doc.metadata.content_hash == ""  # Not yet calculated
        assert isinstance(doc.metadata.extraction_confidence, ExtractionConfidence)
        assert isinstance(doc.metadata.processing_stats, ProcessingStats)
        assert doc.metadata.quality_score == 0.0  # Not yet calculated
    
    def test_enhanced_chunk_creation(self):
        """Test creation of EnhancedChunk with advanced features"""
        chunk = EnhancedChunk(
            content="# Test Header\n\nThis is test content",
            chunk_type=ChunkType.HEADER,
            hierarchy_level=1,
            parent_section="root",
            semantic_topics=["introduction"],
            confidence=0.95
        )
        
        assert chunk.chunk_type == ChunkType.HEADER
        assert chunk.hierarchy_level == 1
        assert chunk.parent_section == "root"
        assert "introduction" in chunk.semantic_topics
        assert chunk.confidence == 0.95
        assert chunk.chunk_id is not None  # Inherited from Chunk
    
    def test_processing_quality_assessment(self, quality_assessor, complex_markdown_content):
        """Test document quality assessment capabilities"""
        doc = EnhancedDocument(complex_markdown_content, "test.md")
        doc.metadata.extraction_confidence.overall = 0.85
        
        quality = quality_assessor.assess_extraction_quality(doc)
        assert quality == ProcessingQuality.GOOD
        
        doc.metadata.extraction_confidence.overall = 0.95
        quality = quality_assessor.assess_extraction_quality(doc)
        assert quality == ProcessingQuality.EXCELLENT
        
        doc.metadata.extraction_confidence.overall = 0.4
        quality = quality_assessor.assess_extraction_quality(doc)
        assert quality == ProcessingQuality.POOR
    
    def test_structure_validation(self, quality_assessor, complex_markdown_content):
        """Test structure extraction validation"""
        doc = EnhancedDocument(complex_markdown_content, "test.md")
        
        # Create some test chunks
        doc.chunks = [
            EnhancedChunk(content="Introduction section", chunk_type=ChunkType.HEADER),
            EnhancedChunk(content="Background content", chunk_type=ChunkType.PARAGRAPH)
        ]
        
        is_valid, issues = quality_assessor.validate_structure_extraction(doc)
        
        # Should identify content loss issue due to simplified chunks
        assert not is_valid
        assert len(issues) > 0
        assert any("content loss" in issue.lower() for issue in issues)
    
    def test_semantic_chunker_topic_detection(self, semantic_chunker):
        """Test topic boundary detection in semantic chunker"""
        content = """
        Introduction to the study
        
        This section provides background information.
        
        Methodology and approach
        
        Our method involves several steps.
        
        Results and findings
        
        The results show significant improvements.
        """
        
        headers = [
            {'line_number': 1, 'title': 'Introduction'},
            {'line_number': 5, 'title': 'Methodology'},
            {'line_number': 9, 'title': 'Results'}
        ]
        
        boundaries = semantic_chunker.detect_topic_boundaries(content, headers)
        
        assert len(boundaries) >= 3  # At least the headers
        assert any('Introduction' in boundary[1] for boundary in boundaries)
    
    def test_hierarchical_chunking(self, semantic_chunker, complex_markdown_content):
        """Test hierarchical chunk creation"""
        doc = EnhancedDocument(complex_markdown_content, "test.md")
        
        # Create a simple hierarchy for testing
        root = DocumentHierarchy(level=0, title="Document Root", content="")
        intro = DocumentHierarchy(level=1, title="Introduction", content="Introduction content", parent=root)
        background = DocumentHierarchy(level=2, title="Background", content="Background content", parent=intro)
        root.children = [intro]
        intro.children = [background]
        
        doc.metadata.hierarchy_tree = root
        
        chunks = semantic_chunker.create_hierarchical_chunks(doc)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, EnhancedChunk) for chunk in chunks)
        
        # Check that hierarchy information is preserved
        header_chunks = [chunk for chunk in chunks if chunk.chunk_type == ChunkType.HEADER]
        assert len(header_chunks) > 0
    
    def test_chunk_type_inference(self, semantic_chunker):
        """Test chunk type inference from content"""
        test_cases = [
            ("# Header", ChunkType.HEADER),
            ("> This is a quote", ChunkType.QUOTE),
            ("```python\ncode here\n```", ChunkType.CODE),
            ("- List item 1\n- List item 2", ChunkType.LIST),
            ("| Col1 | Col2 |\n|------|------|", ChunkType.TABLE),
            ("Regular paragraph text", ChunkType.PARAGRAPH)
        ]
        
        for content, expected_type in test_cases:
            inferred_type = semantic_chunker._infer_chunk_type(content)
            assert inferred_type == expected_type, f"Failed for content: {content}"
    
    def test_advanced_markdown_processing(self, advanced_processor, complex_markdown_content):
        """Test advanced markdown processing with structure extraction"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(complex_markdown_content)
            temp_path = f.name
        
        try:
            doc = advanced_processor.process_file(temp_path, "test.md")
            
            assert isinstance(doc, EnhancedDocument)
            assert doc.document_type == "markdown"
            assert len(doc.metadata.headers) > 0
            
            # Check that chunks are enhanced
            assert all(isinstance(chunk, EnhancedChunk) for chunk in doc.chunks)
            
            # Check semantic topics extraction
            topic_chunks = [chunk for chunk in doc.chunks if chunk.semantic_topics]
            assert len(topic_chunks) > 0
            
        finally:
            os.unlink(temp_path)
    
    def test_quality_score_calculation(self, quality_assessor, complex_markdown_content):
        """Test comprehensive quality score calculation"""
        doc = EnhancedDocument(complex_markdown_content, "test.md")
        
        # Set up metadata for quality calculation
        doc.metadata.extraction_confidence = ExtractionConfidence(
            text_extraction=0.9,
            structure_detection=0.8,
            table_extraction=0.7,
            image_extraction=0.6,
            metadata_extraction=0.8
        )
        doc.metadata.extraction_confidence.calculate_overall()
        
        doc.metadata.headers = [
            {'title': 'Introduction', 'level': 1},
            {'title': 'Background', 'level': 2},
            {'title': 'Results', 'level': 1}
        ]
        
        doc.metadata.processing_stats.total_processing_time = 5.0  # 5 seconds
        
        quality_score = quality_assessor.calculate_quality_score(doc)
        
        assert 0.0 <= quality_score <= 1.0
        assert quality_score > 0.5  # Should be reasonable quality
    
    def test_content_hash_calculation(self, advanced_processor):
        """Test content hash calculation for change detection"""
        content1 = "This is test content"
        content2 = "This is different content"
        
        hash1 = advanced_processor._calculate_content_hash(content1)
        hash2 = advanced_processor._calculate_content_hash(content2)
        hash1_repeat = advanced_processor._calculate_content_hash(content1)
        
        assert hash1 != hash2  # Different content should have different hashes
        assert hash1 == hash1_repeat  # Same content should have same hash
        assert len(hash1) == 16  # Hash should be truncated to 16 characters
    
    def test_processing_statistics_tracking(self, advanced_processor):
        """Test that processing statistics are properly tracked"""
        content = "# Test Document\n\nThis is test content for timing."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            start_time = time.time()
            doc = advanced_processor.process_file(temp_path, "test.txt")
            end_time = time.time()
            
            stats = doc.metadata.processing_stats
            
            assert stats.total_processing_time > 0
            assert stats.total_processing_time <= (end_time - start_time) + 1  # Allow some margin
            assert stats.retry_count >= 0
            assert isinstance(stats.errors_encountered, list)
            
        finally:
            os.unlink(temp_path)
    
    def test_table_schema_creation(self, advanced_processor):
        """Test table schema creation capabilities"""
        # This is a placeholder test since actual table detection depends on docling
        schema = TableSchema(
            headers=["Name", "Age", "City"],
            column_types={"Name": "text", "Age": "integer", "City": "text"},
            row_count=100,
            confidence=0.85,
            extraction_method="docling"
        )
        
        assert len(schema.headers) == 3
        assert schema.column_types["Age"] == "integer"
        assert schema.row_count == 100
        assert schema.confidence == 0.85
        assert schema.extraction_method == "docling"
    
    def test_document_hierarchy_creation(self):
        """Test document hierarchy structure creation"""
        root = DocumentHierarchy(level=0, title="Document", content="")
        
        intro = DocumentHierarchy(level=1, title="Introduction", content="Intro content")
        background = DocumentHierarchy(level=2, title="Background", content="Background content")
        methodology = DocumentHierarchy(level=1, title="Methodology", content="Method content")
        
        root.children = [intro, methodology]
        intro.children = [background]
        
        intro.parent = root
        background.parent = intro
        methodology.parent = root
        
        assert len(root.children) == 2
        assert intro.parent == root
        assert background.parent == intro
        assert len(intro.children) == 1
    
    def test_extraction_confidence_calculation(self):
        """Test extraction confidence calculation and overall score"""
        confidence = ExtractionConfidence(
            text_extraction=0.9,
            structure_detection=0.8,
            table_extraction=0.7,
            image_extraction=0.6,
            metadata_extraction=0.8
        )
        
        overall = confidence.calculate_overall()
        expected = (0.9 + 0.8 + 0.7 + 0.6 + 0.8) / 5
        
        assert abs(overall - expected) < 0.001
        assert confidence.overall == overall
    
    def test_quality_assessment_methods(self, advanced_processor, complex_markdown_content):
        """Test the new quality assessment methods on DocumentProcessor"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(complex_markdown_content)
            temp_path = f.name
        
        try:
            doc = advanced_processor.process_file(temp_path, "test.md")
            
            # Test quality assessment
            quality, issues = advanced_processor.assess_document_quality(doc)
            assert isinstance(quality, ProcessingQuality)
            assert isinstance(issues, list)
            
            # Test stats retrieval
            stats = advanced_processor.get_processing_stats(doc)
            assert isinstance(stats, ProcessingStats)
            
            # Test confidence retrieval
            confidence = advanced_processor.get_extraction_confidence(doc)
            assert isinstance(confidence, ExtractionConfidence)
            
        finally:
            os.unlink(temp_path)
    
    def test_backward_compatibility(self, advanced_processor):
        """Test that enhanced processor maintains backward compatibility"""
        content = "This is a test document for backward compatibility."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            # Process file should still return an EnhancedDocument
            doc = advanced_processor.process_file(temp_path, "test.txt")
            
            # But it should still have all the basic Document properties
            assert hasattr(doc, 'content')
            assert hasattr(doc, 'filename')
            assert hasattr(doc, 'id')
            assert hasattr(doc, 'chunks')
            
            # And chunks should still be usable as basic chunks
            if doc.chunks:
                chunk = doc.chunks[0]
                assert hasattr(chunk, 'content')
                assert hasattr(chunk, 'start_index')
                assert hasattr(chunk, 'end_index')
                assert hasattr(chunk, 'chunk_id')
            
        finally:
            os.unlink(temp_path)
    
    def test_large_document_scaling(self, advanced_processor):
        """Test that advanced features work with large documents"""
        # Create a large document (simulating 10k+ document processing scale)
        large_content = "# Large Document\n\n"
        
        # Add multiple sections with substantial content
        for i in range(100):
            large_content += f"## Section {i+1}\n\n"
            large_content += "This is substantial content for testing scaling. " * 50
            large_content += "\n\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(large_content)
            temp_path = f.name
        
        try:
            start_time = time.time()
            doc = advanced_processor.process_file(temp_path, "large_test.md")
            processing_time = time.time() - start_time
            
            # Verify document was processed successfully
            assert isinstance(doc, EnhancedDocument)
            assert len(doc.content) > 50000  # Should be substantial
            assert len(doc.chunks) > 50  # Should create many chunks
            
            # Verify processing completed in reasonable time (less than 30 seconds)
            assert processing_time < 30, f"Processing took {processing_time:.2f} seconds"
            
            # Verify quality metrics are reasonable
            quality, issues = advanced_processor.assess_document_quality(doc)
            assert quality != ProcessingQuality.POOR
            
        finally:
            os.unlink(temp_path)