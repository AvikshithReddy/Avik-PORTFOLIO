"""
Pydantic schemas for API request/response models
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    query: str = Field(..., min_length=1, max_length=2000, description="User query")
    session_id: Optional[str] = Field(None, description="Session identifier for conversation context")
    include_sources: bool = Field(True, description="Whether to include source citations")


class SourceReference(BaseModel):
    """Reference to a source document"""
    source_type: str = Field(..., description="Type of source: resume, portfolio, markdown, github")
    source_name: str = Field(..., description="Name of the source")
    locator: str = Field(..., description="File path or URL to the source")
    snippet: str = Field(..., description="Relevant snippet from the source")
    relevance_score: Optional[float] = Field(None, description="Relevance score (0-1)")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="Generated response text")
    session_id: str = Field(..., description="Session identifier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the response")
    sources: List[SourceReference] = Field(default_factory=list, description="Source citations")


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    rag_index_loaded: bool = Field(..., description="Whether RAG index is loaded")
    total_documents: int = Field(0, description="Number of documents in index")


class IngestRequest(BaseModel):
    """Request model for ingestion endpoint"""
    sources: List[str] = Field(
        default=["portfolio", "resume", "markdown", "github"],
        description="List of sources to ingest"
    )
    force_rebuild: bool = Field(False, description="Force rebuild even if index exists")


class IngestResponse(BaseModel):
    """Response model for ingestion endpoint"""
    status: str = Field(..., description="Ingestion status")
    documents_processed: int = Field(..., description="Number of documents processed")
    chunks_created: int = Field(..., description="Number of chunks created")
    embeddings_generated: int = Field(..., description="Number of embeddings generated")
    sources_ingested: List[str] = Field(..., description="List of sources successfully ingested")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
