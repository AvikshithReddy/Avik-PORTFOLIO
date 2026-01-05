"""
Main FastAPI application for the portfolio chatbot
"""

import uuid
from typing import Dict
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SourceReference
)
from app.llm.openai_client import OpenAIClient
from app.rag import RAGIndex, DocumentBuilder, SYSTEM_PROMPT, build_rag_prompt, build_clarification_prompt
from app.utils.logging import app_logger
from app.utils.text import truncate_text

# Global instances
openai_client: OpenAIClient = None
rag_index: RAGIndex = None
doc_builder: DocumentBuilder = None

# Session storage (in production, use Redis or similar)
sessions: Dict[str, list] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global openai_client, rag_index, doc_builder
    
    # Startup
    app_logger.info("Starting Portfolio Chatbot API...")
    
    # Ensure directories exist
    settings.ensure_directories()
    
    # Initialize OpenAI client
    openai_client = OpenAIClient(
        api_key=settings.OPENAI_API_KEY,
        chat_model=settings.CHAT_MODEL,
        embedding_model=settings.EMBEDDING_MODEL
    )
    
    # Initialize RAG index
    rag_index = RAGIndex(settings.RAG_INDEX_DIR)
    
    # Try to load existing index
    loaded = rag_index.load_index()
    if loaded:
        app_logger.info("Loaded existing RAG index")
    else:
        app_logger.warning("No existing RAG index found. Run /api/ingest to build index.")
    
    # Initialize document builder
    doc_builder = DocumentBuilder(openai_client, rag_index)
    
    app_logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    app_logger.info("Shutting down Portfolio Chatbot API...")


# Create FastAPI app
app = FastAPI(
    title="Portfolio Chatbot API",
    description="Intelligent chatbot for Avikshith Reddy's portfolio with RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    stats = rag_index.get_stats() if rag_index else {"loaded": False}
    
    return HealthResponse(
        status="ok",
        version="1.0.0",
        rag_index_loaded=stats.get("loaded", False),
        total_documents=stats.get("total_chunks", 0)
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with RAG-powered responses
    
    Args:
        request: Chat request with query and optional session_id
    
    Returns:
        Chat response with answer and sources
    """
    if not rag_index or not rag_index.loaded:
        raise HTTPException(
            status_code=503,
            detail="RAG index not loaded. Please run /api/ingest first."
        )
    
    try:
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())
        if session_id not in sessions:
            sessions[session_id] = []
        
        # Get query embedding
        query_embedding = openai_client.create_embedding(request.query)
        
        # Search for relevant chunks
        search_results = rag_index.search(
            query_embedding,
            top_k=settings.RAG_TOP_K
        )
        
        # Calculate confidence from top result
        confidence = search_results[0][1] if search_results else 0.0
        
        # Build context chunks
        context_chunks = []
        sources = []
        
        for metadata, score in search_results:
            # Add to context
            context_chunks.append({
                "text": metadata.get("text", ""),
                "metadata": metadata
            })
            
            # Build source reference
            if request.include_sources:
                source = SourceReference(
                    source_type=metadata.get("source_type", "unknown"),
                    source_name=metadata.get("source_name", "unknown"),
                    locator=metadata.get("locator", ""),
                    snippet=truncate_text(metadata.get("text", ""), max_length=150),
                    relevance_score=round(score, 3)
                )
                sources.append(source)
        
        # Build messages for chat completion
        if confidence >= settings.RAG_CONFIDENCE_THRESHOLD:
            user_prompt = build_rag_prompt(request.query, context_chunks)
        else:
            user_prompt = build_clarification_prompt(request.query, search_results)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # Add conversation history (last 3 exchanges)
        conversation_history = sessions[session_id][-6:]  # Last 3 Q&A pairs
        for msg in conversation_history:
            messages.insert(-1, msg)  # Insert before current user message
        
        # Generate response
        response_text = openai_client.chat_completion(
            messages=messages,
            temperature=settings.CHAT_TEMPERATURE,
            max_tokens=settings.CHAT_MAX_TOKENS
        )
        
        # Update session history
        sessions[session_id].append({"role": "user", "content": request.query})
        sessions[session_id].append({"role": "assistant", "content": response_text})
        
        # Keep only recent history
        if len(sessions[session_id]) > 20:
            sessions[session_id] = sessions[session_id][-20:]
        
        app_logger.info(f"Chat response generated (confidence: {confidence:.3f})")
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            confidence=round(confidence, 3),
            sources=sources if request.include_sources else []
        )
    
    except Exception as e:
        app_logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")


async def verify_admin_key(x_admin_key: str = Header(None)):
    """Verify admin key for protected endpoints"""
    if settings.ADMIN_INGEST_KEY and x_admin_key != settings.ADMIN_INGEST_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return True


@app.post("/api/ingest", response_model=IngestResponse, dependencies=[Depends(verify_admin_key)])
async def ingest_documents(request: IngestRequest):
    """
    Ingest and build RAG index from sources
    
    Protected endpoint requiring X-Admin-Key header
    
    Args:
        request: Ingest request with sources and options
    
    Returns:
        Ingestion statistics
    """
    if not doc_builder:
        raise HTTPException(status_code=503, detail="Document builder not initialized")
    
    try:
        app_logger.info(f"Starting ingestion: sources={request.sources}, force_rebuild={request.force_rebuild}")
        
        result = doc_builder.build_index(
            sources=request.sources,
            force_rebuild=request.force_rebuild
        )
        
        return IngestResponse(
            status=result["status"],
            documents_processed=result["documents_processed"],
            chunks_created=result["chunks_created"],
            embeddings_generated=result["embeddings_generated"],
            sources_ingested=result["sources_ingested"],
            errors=result.get("errors", [])
        )
    
    except Exception as e:
        app_logger.error(f"Ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Portfolio Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/api/chat",
            "ingest": "/api/ingest (admin only)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL
    )
