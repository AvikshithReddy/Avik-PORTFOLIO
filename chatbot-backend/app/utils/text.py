"""
Utility functions for text processing and chunking
"""

import re
from typing import List, Dict, Any


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove multiple whitespaces
    text = re.sub(r'\s+', ' ', text)
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    metadata: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks with metadata
    
    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks
        metadata: Metadata to attach to each chunk
    
    Returns:
        List of chunk dictionaries with text and metadata
    """
    if metadata is None:
        metadata = {}
    
    # Clean the text first
    text = clean_text(text)
    
    if len(text) <= chunk_size:
        return [{
            "text": text,
            "metadata": {**metadata, "chunk_index": 0, "total_chunks": 1}
        }]
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        # Calculate end position
        end = start + chunk_size
        
        # If not the last chunk, try to break at sentence or word boundary
        if end < len(text):
            # Look for sentence boundary (. ! ? followed by space)
            sentence_break = text.rfind('. ', start, end)
            if sentence_break == -1:
                sentence_break = text.rfind('! ', start, end)
            if sentence_break == -1:
                sentence_break = text.rfind('? ', start, end)
            
            if sentence_break > start:
                end = sentence_break + 1
            else:
                # Look for word boundary
                space_pos = text.rfind(' ', start, end)
                if space_pos > start:
                    end = space_pos
        
        # Extract chunk
        chunk_text_content = text[start:end].strip()
        
        if chunk_text_content:
            chunks.append({
                "text": chunk_text_content,
                "metadata": {
                    **metadata,
                    "text": chunk_text_content,  # Store chunk text for snippet extraction
                    "chunk_index": chunk_index,
                    "char_start": start,
                    "char_end": end
                }
            })
            chunk_index += 1
        
        # Move start position with overlap
        start = end - chunk_overlap if end < len(text) else end
    
    # Add total_chunks to metadata
    for chunk in chunks:
        chunk["metadata"]["total_chunks"] = len(chunks)
    
    return chunks


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def extract_code_blocks(text: str) -> List[str]:
    """Extract code blocks from markdown text"""
    pattern = r'```[\w]*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def count_tokens_estimate(text: str) -> int:
    """
    Rough estimate of token count
    (Real implementation would use tiktoken)
    """
    # Rough estimate: ~4 characters per token
    return len(text) // 4
