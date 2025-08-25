"""
Enhanced document processor with support for PDF and Markdown using docling.
Maintains backward compatibility with the original TextProcessor.
"""
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import os
import logging
from pathlib import Path

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logging.warning("Docling not available. PDF/Markdown processing will use fallback methods.")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import markdown
    from markdown.extensions import codehilite, tables, toc
    import frontmatter
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

from .models import Document, Chunk
from .text_processor import TextProcessor


class DocumentMetadata:
    """Enhanced metadata for documents"""
    
    def __init__(self):
        self.title: Optional[str] = None
        self.author: Optional[str] = None
        self.creation_date: Optional[str] = None
        self.modification_date: Optional[str] = None
        self.page_count: Optional[int] = None
        self.word_count: Optional[int] = None
        self.document_type: Optional[str] = None
        self.headers: List[Dict[str, Any]] = []
        self.tables: List[Dict[str, Any]] = []
        self.links: List[str] = []
        self.language: Optional[str] = None


class EnhancedDocument(Document):
    """Document with enhanced metadata and structure"""
    
    def __init__(self, content: str, filename: str, **kwargs):
        super().__init__(content, filename, **kwargs)
        self.metadata: DocumentMetadata = DocumentMetadata()
        self.structured_content: List[Dict[str, Any]] = []
        self.document_type: str = self._infer_type(filename)
    
    def _infer_type(self, filename: str) -> str:
        """Infer document type from filename"""
        ext = Path(filename).suffix.lower()
        type_mapping = {
            '.pdf': 'pdf',
            '.md': 'markdown',
            '.markdown': 'markdown',
            '.txt': 'text',
            '.docx': 'word',
            '.doc': 'word',
            '.html': 'html',
            '.htm': 'html'
        }
        return type_mapping.get(ext, 'unknown')


class DocumentProcessor:
    """Enhanced document processor with PDF and Markdown support"""
    
    def __init__(self, use_docling: bool = True):
        self.use_docling = use_docling and DOCLING_AVAILABLE
        self.text_processor = TextProcessor()  # Fallback processor
        
        if self.use_docling:
            self.doc_converter = DocumentConverter()
        
        logging.info(f"DocumentProcessor initialized. Docling: {self.use_docling}")
    
    def process_file(self, file_path: str, filename: Optional[str] = None) -> EnhancedDocument:
        """Process a file and return enhanced document"""
        if filename is None:
            filename = os.path.basename(file_path)
        
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.pdf':
                return self._process_pdf(file_path, filename)
            elif file_ext in ['.md', '.markdown']:
                return self._process_markdown(file_path, filename)
            elif file_ext in ['.txt']:
                return self._process_text(file_path, filename)
            else:
                # Try docling for other formats
                if self.use_docling:
                    return self._process_with_docling(file_path, filename)
                else:
                    raise ValueError(f"Unsupported file format: {file_ext}")
        
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
            # Fallback to text processing
            return self._process_text(file_path, filename)
    
    def process_bytes(self, file_content: bytes, filename: str) -> EnhancedDocument:
        """Process file content from bytes"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            result = self.process_file(tmp_path, filename)
        finally:
            # Cleanup
            os.unlink(tmp_path)
        
        return result
    
    def _process_pdf(self, file_path: str, filename: str) -> EnhancedDocument:
        """Process PDF file"""
        if self.use_docling:
            return self._process_pdf_with_docling(file_path, filename)
        elif PYPDF2_AVAILABLE:
            return self._process_pdf_with_pypdf2(file_path, filename)
        else:
            raise ValueError("No PDF processing library available")
    
    def _process_pdf_with_docling(self, file_path: str, filename: str) -> EnhancedDocument:
        """Process PDF using docling"""
        try:
            # Convert PDF to structured format
            result = self.doc_converter.convert(file_path)
            
            # Extract text content
            content = result.document.export_to_markdown()
            
            # Create enhanced document
            doc = EnhancedDocument(content, filename)
            
            # Extract metadata
            if hasattr(result.document, 'meta'):
                meta = result.document.meta
                doc.metadata.title = getattr(meta, 'title', None)
                doc.metadata.author = getattr(meta, 'author', None)
                doc.metadata.creation_date = getattr(meta, 'creation_date', None)
                doc.metadata.page_count = getattr(meta, 'page_count', None)
            
            # Extract structured content
            if hasattr(result.document, 'body'):
                doc.structured_content = self._extract_structure(result.document.body)
            
            # Generate semantic chunks
            doc.chunks = self._create_semantic_chunks(doc)
            
            return doc
            
        except Exception as e:
            logging.error(f"Docling PDF processing failed: {e}")
            # Fallback to PyPDF2
            return self._process_pdf_with_pypdf2(file_path, filename)
    
    def _process_pdf_with_pypdf2(self, file_path: str, filename: str) -> EnhancedDocument:
        """Process PDF using PyPDF2 (fallback)"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from all pages
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n\n"
                
                # Create document
                doc = EnhancedDocument(text_content.strip(), filename)
                doc.metadata.page_count = len(pdf_reader.pages)
                
                # Try to extract metadata
                if pdf_reader.metadata:
                    doc.metadata.title = pdf_reader.metadata.get('/Title')
                    doc.metadata.author = pdf_reader.metadata.get('/Author')
                    doc.metadata.creation_date = pdf_reader.metadata.get('/CreationDate')
                
                # Generate standard chunks (fallback)
                doc.chunks = self.text_processor.chunk_text(doc.content)
                
                return doc
                
        except Exception as e:
            logging.error(f"PyPDF2 processing failed: {e}")
            raise
    
    def _process_markdown(self, file_path: str, filename: str) -> EnhancedDocument:
        """Process Markdown file"""
        try:
            # Read file with frontmatter support
            if MARKDOWN_AVAILABLE:
                with open(file_path, 'r', encoding='utf-8') as f:
                    post = frontmatter.load(f)
                
                # Convert markdown to text (keeping structure)
                md = markdown.Markdown(extensions=['toc', 'tables', 'codehilite'])
                html_content = md.convert(post.content)
                
                # For now, keep markdown as-is for better chunking
                content = post.content
                
                # Create enhanced document
                doc = EnhancedDocument(content, filename)
                
                # Extract frontmatter metadata
                for key, value in post.metadata.items():
                    if key == 'title':
                        doc.metadata.title = value
                    elif key == 'author':
                        doc.metadata.author = value
                    elif key == 'date':
                        doc.metadata.creation_date = str(value)
                
                # Extract headers for structure
                doc.metadata.headers = self._extract_md_headers(content)
                
                # Generate semantic chunks
                doc.chunks = self._create_semantic_chunks(doc)
                
                return doc
            else:
                # Fallback to plain text processing
                return self._process_text(file_path, filename)
                
        except Exception as e:
            logging.error(f"Markdown processing failed: {e}")
            return self._process_text(file_path, filename)
    
    def _process_text(self, file_path: str, filename: str) -> EnhancedDocument:
        """Process plain text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        doc = EnhancedDocument(content, filename)
        doc.chunks = self.text_processor.chunk_text(content)
        
        return doc
    
    def _process_with_docling(self, file_path: str, filename: str) -> EnhancedDocument:
        """Process any supported format with docling"""
        try:
            result = self.doc_converter.convert(file_path)
            content = result.document.export_to_markdown()
            
            doc = EnhancedDocument(content, filename)
            doc.chunks = self._create_semantic_chunks(doc)
            
            return doc
        except Exception as e:
            logging.error(f"Docling general processing failed: {e}")
            # Final fallback
            return self._process_text(file_path, filename)
    
    def _extract_md_headers(self, content: str) -> List[Dict[str, Any]]:
        """Extract headers from markdown content"""
        headers = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                headers.append({
                    'level': level,
                    'title': title,
                    'line_number': i + 1
                })
        
        return headers
    
    def _extract_structure(self, body) -> List[Dict[str, Any]]:
        """Extract structured content from docling result"""
        structured = []
        
        # This is a placeholder - actual implementation depends on docling's data structure
        # The goal is to preserve document hierarchy and structure
        
        return structured
    
    def _create_semantic_chunks(self, doc: EnhancedDocument) -> List[Chunk]:
        """Create semantic chunks based on document structure"""
        if doc.document_type == 'markdown' and doc.metadata.headers:
            return self._chunk_by_headers(doc)
        else:
            # Fallback to standard chunking
            return self.text_processor.chunk_text(doc.content)
    
    def _chunk_by_headers(self, doc: EnhancedDocument) -> List[Chunk]:
        """Chunk markdown content by headers"""
        chunks = []
        content_lines = doc.content.split('\n')
        
        current_section = []
        current_header = None
        
        for i, line in enumerate(content_lines):
            line_stripped = line.strip()
            
            # Check if this is a header
            is_header = line_stripped.startswith('#')
            
            if is_header and current_section:
                # Save current section as chunk
                section_content = '\n'.join(current_section).strip()
                if section_content:
                    chunk = Chunk(
                        content=section_content,
                        start_index=max(0, i - len(current_section)),
                        end_index=i
                    )
                    chunks.append(chunk)
                
                # Start new section
                current_section = [line]
                current_header = line_stripped
            else:
                current_section.append(line)
        
        # Add final section
        if current_section:
            section_content = '\n'.join(current_section).strip()
            if section_content:
                chunk = Chunk(
                    content=section_content,
                    start_index=len(content_lines) - len(current_section),
                    end_index=len(content_lines)
                )
                chunks.append(chunk)
        
        # If no chunks created, fallback to standard chunking
        if not chunks:
            chunks = self.text_processor.chunk_text(doc.content)
        
        return chunks
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        formats = ['.txt']
        
        if self.use_docling:
            formats.extend(['.pdf', '.md', '.markdown', '.docx', '.doc', '.html'])
        elif PYPDF2_AVAILABLE:
            formats.append('.pdf')
        
        if MARKDOWN_AVAILABLE:
            formats.extend(['.md', '.markdown'])
        
        return formats
    
    def is_supported_format(self, filename: str) -> bool:
        """Check if file format is supported"""
        ext = Path(filename).suffix.lower()
        return ext in self.get_supported_formats()