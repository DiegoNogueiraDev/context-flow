from typing import List
import re
from .models import Document, Chunk


class TextProcessor:
    """Handles text processing, chunking, and document creation"""
    
    def create_document(self, content: str, filename: str) -> Document:
        """Create a Document from text content"""
        return Document(content=content, filename=filename)
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[Chunk]:
        """Split text into overlapping chunks"""
        if not text or not text.strip():
            return []
        
        chunks = []
        text = text.strip()
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Find word boundary if we're not at the end
            if end < len(text):
                # Look backwards for sentence end or word boundary
                for i in range(end, max(start + chunk_size // 2, start + 1), -1):
                    if text[i-1] in '.!?':
                        end = i
                        break
                    elif text[i-1] in ' \t\n':
                        end = i
                        break
            
            chunk_content = text[start:end].strip()
            if chunk_content:
                chunk = Chunk(
                    content=chunk_content,
                    start_index=start,
                    end_index=end
                )
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap if end - overlap > start else end
            
            if start >= len(text):
                break
        
        return chunks
    
    def process_upload(self, file_content: bytes, filename: str) -> Document:
        """Process uploaded file content and create document with chunks"""
        content = file_content.decode('utf-8')
        document = self.create_document(content, filename)
        document.chunks = self.chunk_text(content)
        return document