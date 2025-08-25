import pytest
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.document_processor import DocumentProcessor, EnhancedDocument, DocumentMetadata


class TestDocumentProcessor:
    
    @pytest.fixture
    def document_processor(self):
        """Create DocumentProcessor instance for testing"""
        return DocumentProcessor(use_docling=False)  # Use fallback for testing
    
    @pytest.fixture
    def sample_text_content(self):
        """Sample text content for testing"""
        return """
        # Introduction to Machine Learning

        Machine learning is a subset of artificial intelligence (AI) that enables computers 
        to learn and improve from experience without being explicitly programmed.

        ## Types of Machine Learning

        ### Supervised Learning
        Supervised learning uses labeled training data to learn a mapping from inputs to outputs.

        ### Unsupervised Learning  
        Unsupervised learning finds hidden patterns in data without labeled examples.

        ### Reinforcement Learning
        Reinforcement learning learns through interaction with an environment.
        """
    
    @pytest.fixture
    def sample_markdown_file(self, sample_text_content):
        """Create temporary markdown file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(sample_text_content)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def sample_text_file(self):
        """Create temporary text file for testing"""
        content = "This is a simple text document for testing purposes. It contains multiple sentences."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_initialization(self, document_processor):
        """Test DocumentProcessor initialization"""
        assert document_processor is not None
        assert hasattr(document_processor, 'text_processor')
        assert hasattr(document_processor, 'use_docling')
    
    def test_supported_formats(self, document_processor):
        """Test supported file format checking"""
        supported = document_processor.get_supported_formats()
        
        assert '.txt' in supported
        assert isinstance(supported, list)
        assert len(supported) > 0
        
        # Test format checking
        assert document_processor.is_supported_format('test.txt')
        assert document_processor.is_supported_format('test.md') or not document_processor.use_docling
        assert not document_processor.is_supported_format('test.xyz')
    
    def test_process_text_file(self, document_processor, sample_text_file):
        """Test processing plain text file"""
        result = document_processor.process_file(sample_text_file, "test.txt")
        
        assert isinstance(result, EnhancedDocument)
        assert result.filename == "test.txt"
        assert result.document_type == "text"
        assert len(result.content) > 0
        assert len(result.chunks) > 0
        assert isinstance(result.metadata, DocumentMetadata)
    
    def test_process_markdown_file(self, document_processor, sample_markdown_file):
        """Test processing markdown file"""
        result = document_processor.process_file(sample_markdown_file, "test.md")
        
        assert isinstance(result, EnhancedDocument)
        assert result.filename == "test.md"
        assert result.document_type == "markdown"
        assert "Machine Learning" in result.content
        assert len(result.chunks) > 0
        
        # Check if headers are extracted
        if hasattr(result.metadata, 'headers') and result.metadata.headers:
            header_titles = [h['title'] for h in result.metadata.headers]
            assert any("Machine Learning" in title for title in header_titles)
    
    def test_process_bytes(self, document_processor):
        """Test processing document from bytes"""
        content = b"This is test content from bytes."
        filename = "test_bytes.txt"
        
        result = document_processor.process_bytes(content, filename)
        
        assert isinstance(result, EnhancedDocument)
        assert result.filename == filename
        assert "test content from bytes" in result.content
        assert len(result.chunks) > 0
    
    def test_enhanced_document_creation(self):
        """Test EnhancedDocument creation and properties"""
        content = "Test document content"
        filename = "test.pdf"
        
        doc = EnhancedDocument(content, filename)
        
        assert doc.content == content
        assert doc.filename == filename
        assert doc.document_type == "pdf"
        assert isinstance(doc.metadata, DocumentMetadata)
        assert doc.id is not None
        assert isinstance(doc.structured_content, list)
    
    def test_document_metadata(self):
        """Test DocumentMetadata functionality"""
        metadata = DocumentMetadata()
        
        assert metadata.title is None
        assert metadata.author is None
        assert isinstance(metadata.headers, list)
        assert isinstance(metadata.tables, list)
        assert isinstance(metadata.links, list)
        
        # Test setting metadata
        metadata.title = "Test Document"
        metadata.author = "Test Author"
        
        assert metadata.title == "Test Document"
        assert metadata.author == "Test Author"
    
    def test_markdown_header_extraction(self, document_processor):
        """Test markdown header extraction"""
        md_content = """
        # Main Title
        
        Some content here.
        
        ## Section 1
        
        More content.
        
        ### Subsection 1.1
        
        Even more content.
        """
        
        headers = document_processor._extract_md_headers(md_content)
        
        assert len(headers) == 3
        assert headers[0]['level'] == 1
        assert headers[0]['title'] == "Main Title"
        assert headers[1]['level'] == 2
        assert headers[1]['title'] == "Section 1"
        assert headers[2]['level'] == 3
        assert headers[2]['title'] == "Subsection 1.1"
    
    def test_semantic_chunking_by_headers(self, document_processor, sample_text_content):
        """Test semantic chunking based on markdown headers"""
        doc = EnhancedDocument(sample_text_content, "test.md")
        doc.metadata.headers = document_processor._extract_md_headers(sample_text_content)
        
        chunks = document_processor._chunk_by_headers(doc)
        
        assert len(chunks) > 0
        assert all(hasattr(chunk, 'content') for chunk in chunks)
        assert all(hasattr(chunk, 'chunk_id') for chunk in chunks)
        
        # Check that chunks contain meaningful content
        chunk_contents = [chunk.content for chunk in chunks]
        assert any("Machine learning" in content for content in chunk_contents)
        assert any("Supervised Learning" in content for content in chunk_contents)
    
    def test_fallback_chunking(self, document_processor):
        """Test fallback to standard chunking when no headers found"""
        content = "This is plain text without headers. Just regular sentences for testing chunking behavior."
        doc = EnhancedDocument(content, "test.txt")
        doc.metadata.headers = []  # No headers
        
        chunks = document_processor._chunk_by_headers(doc)
        
        assert len(chunks) > 0
        assert all(hasattr(chunk, 'content') for chunk in chunks)
    
    def test_unsupported_format_handling(self, document_processor):
        """Test handling of unsupported file formats"""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            f.write(b"test content")
            temp_path = f.name
        
        try:
            # Should raise error for unsupported format
            with pytest.raises(ValueError):
                document_processor.process_file(temp_path, "test.xyz")
        finally:
            os.unlink(temp_path)
    
    def test_empty_file_handling(self, document_processor):
        """Test handling of empty files"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name
        
        try:
            result = document_processor.process_file(temp_path, "empty.txt")
            
            # Should handle empty files gracefully
            assert isinstance(result, EnhancedDocument)
            assert result.filename == "empty.txt"
            # Chunks might be empty or have minimal content
            
        finally:
            os.unlink(temp_path)
    
    def test_large_document_handling(self, document_processor):
        """Test handling of large documents"""
        # Create a large text document
        large_content = "This is a test sentence. " * 1000  # ~25KB of text
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(large_content)
            temp_path = f.name
        
        try:
            result = document_processor.process_file(temp_path, "large.txt")
            
            assert isinstance(result, EnhancedDocument)
            assert len(result.content) > 20000  # Should contain the full content
            assert len(result.chunks) > 1  # Should be split into multiple chunks
            
            # Verify chunk overlap and boundaries
            if len(result.chunks) > 1:
                # Check that chunks have reasonable sizes
                chunk_sizes = [len(chunk.content) for chunk in result.chunks]
                assert all(size > 0 for size in chunk_sizes)
                assert max(chunk_sizes) <= 1000  # Reasonable chunk size limit
            
        finally:
            os.unlink(temp_path)
    
    def test_error_recovery(self, document_processor):
        """Test error recovery and fallback behavior"""
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            document_processor.process_file("non_existent_file.txt", "test.txt")
        
        # Test with corrupted/invalid content should not crash
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as f:
            f.write(b'\x00\x01\x02\x03\xff\xfe\xfd')  # Binary garbage
            temp_path = f.name
        
        try:
            # Should handle gracefully, possibly falling back to text processing
            result = document_processor.process_file(temp_path, "corrupted.txt")
            assert isinstance(result, EnhancedDocument)
            
        except Exception as e:
            # If it does raise an exception, it should be a meaningful one
            assert isinstance(e, (ValueError, UnicodeDecodeError))
        finally:
            os.unlink(temp_path)
    
    def test_metadata_preservation(self, document_processor):
        """Test that document metadata is properly preserved"""
        content_with_frontmatter = """---
title: Test Document
author: Test Author
date: 2024-01-01
---

# Test Document

This is the main content of the document.
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content_with_frontmatter)
            temp_path = f.name
        
        try:
            result = document_processor.process_file(temp_path, "test_with_metadata.md")
            
            assert isinstance(result, EnhancedDocument)
            
            # Check if metadata was extracted (depends on markdown library availability)
            if hasattr(result.metadata, 'title') and result.metadata.title:
                assert result.metadata.title == "Test Document"
            if hasattr(result.metadata, 'author') and result.metadata.author:
                assert result.metadata.author == "Test Author"
            
        finally:
            os.unlink(temp_path)