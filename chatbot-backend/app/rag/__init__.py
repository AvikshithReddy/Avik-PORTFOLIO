"""
RAG package for retrieval-augmented generation
"""

from .index import RAGIndex
from .build_docs import DocumentBuilder
from .prompts import SYSTEM_PROMPT, build_rag_prompt, build_clarification_prompt

__all__ = ['RAGIndex', 'DocumentBuilder', 'SYSTEM_PROMPT', 'build_rag_prompt', 'build_clarification_prompt']
