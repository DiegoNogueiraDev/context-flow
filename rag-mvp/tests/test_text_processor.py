import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.text_processor import TextProcessor, Document, Chunk


class TestTextProcessor:
    
    def test_create_document_from_text(self):
        """Test creating a Document from text content"""
        processor = TextProcessor()
        content = "This is a test document with some content."
        filename = "test.txt"
        
        document = processor.create_document(content, filename)
        
        assert isinstance(document, Document)
        assert document.content == content
        assert document.filename == filename
        assert document.id is not None
        assert len(document.id) > 0
    
    def test_chunk_text_basic(self):
        """Test basic text chunking functionality"""
        processor = TextProcessor()
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        
        chunks = processor.chunk_text(text, chunk_size=50)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(len(chunk.content) <= 50 for chunk in chunks)
    
    def test_chunk_text_with_overlap(self):
        """Test text chunking with overlap"""
        processor = TextProcessor()
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        
        chunks = processor.chunk_text(text, chunk_size=30, overlap=10)
        
        assert len(chunks) >= 2
        # Verify overlap exists between consecutive chunks
        if len(chunks) > 1:
            chunk1_end = chunks[0].content[-10:]
            chunk2_start = chunks[1].content[:10]
            # Should have some overlap (not exact due to word boundaries)
            assert len(set(chunk1_end.split()) & set(chunk2_start.split())) > 0
    
    def test_chunk_empty_text(self):
        """Test chunking empty or whitespace text"""
        processor = TextProcessor()
        
        chunks = processor.chunk_text("")
        assert chunks == []
        
        chunks = processor.chunk_text("   \n\t  ")
        assert chunks == []
    
    def test_chunk_text_preserves_meaning(self):
        """Test that chunking preserves semantic boundaries"""
        processor = TextProcessor()
        text = """
        Machine learning is a subset of artificial intelligence. 
        It focuses on algorithms that can learn from data.
        Deep learning is a subset of machine learning.
        It uses neural networks with multiple layers.
        """
        
        chunks = processor.chunk_text(text, chunk_size=100)
        
        # Verify each chunk has meaningful content
        for chunk in chunks:
            assert chunk.content.strip()
            assert len(chunk.content.strip()) > 10  # Not just fragments
    
    def test_process_uploaded_file_content(self):
        """Test processing file content as would come from upload"""
        processor = TextProcessor()
        file_content = b"This is file content as bytes from upload."
        filename = "uploaded.txt"
        
        document = processor.process_upload(file_content, filename)
        
        assert isinstance(document, Document)
        assert document.content == "This is file content as bytes from upload."
        assert document.filename == filename
        assert document.chunks is not None
        assert len(document.chunks) > 0