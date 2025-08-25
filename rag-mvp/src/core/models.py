from dataclasses import dataclass
from typing import List, Optional
import uuid


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    content: str
    start_index: int = 0
    end_index: int = 0
    chunk_id: str = None
    
    def __post_init__(self):
        if self.chunk_id is None:
            self.chunk_id = str(uuid.uuid4())


@dataclass
class Document:
    """Represents a document with its chunks"""
    content: str
    filename: str
    id: str = None
    chunks: Optional[List[Chunk]] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.chunks is None:
            self.chunks = []