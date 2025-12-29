"""
🚀 ADVANCED AI PORTFOLIO CHATBOT - PRODUCTION SERVER
With: GitHub Integration + Resume Processing + RAG + Agents + Memory + Reasoning

Implements:
- Multi-source data integration (GitHub, Resume, Portfolio)
- Advanced RAG with semantic search and embeddings
- Agent-based reasoning loop
- Sliding window memory and conversation summarization
- Vector database for efficient retrieval
- Production-ready FastAPI server
"""

import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import custom modules
from openai import OpenAI
from github_data_fetcher import GitHubDataFetcher
from resume_processor import ResumeProcessor
from memory_system import ContextManager, SlidingWindowMemory
from agent import PortfolioAgent, AgentTool
from vector_db import VectorStore, SemanticRetrieval, DocumentIndexer
from embeddings_manager import EmbeddingsManager

# ==================== FASTAPI SETUP ====================

app = FastAPI(
    title="🎯 Advanced AI Portfolio Chatbot",
    description="Multi-source RAG chatbot with agent reasoning and memory",
    version="3.0.0"
)

# CORS Configuration
cors_origins = os.getenv("CORS_ORIGINS", '["*"]')
try:
    cors_origins = json.loads(cors_origins)
except:
    cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== CONFIGURATION ====================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "avikshithreddy")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

RESUME_PATH = os.getenv("RESUME_PATH", "./avikshithReddy_resume.pdf")
PORTFOLIO_DATA_PATH = os.getenv("PORTFOLIO_DATA_PATH", "./portfolio_data.json")

MAX_CONVERSATION_TURNS = int(os.getenv("MAX_CONVERSATION_TURNS", "10"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "3000"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

MAX_REASONING_STEPS = int(os.getenv("MAX_REASONING_STEPS", "5"))

# Validate API key
if not OPENAI_API_KEY:
    logger.warning("⚠️  OPENAI_API_KEY not set. Chatbot will have limited functionality.")
    openai_client = None
else:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("✅ OpenAI client initialized")

# ==================== DATA MODELS ====================

class ChatRequest(BaseModel):
    """Chat request model"""
    query: str
    session_id: Optional[str] = None
    include_reasoning: bool = False
    include_sources: bool = True


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    session_id: str
    sources: List[str] = []
    confidence: float = 0.0
    reasoning: Optional[List[Dict]] = None
    metadata: Dict = {}


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    initialized: bool
    components: Dict


# ==================== GLOBAL STATE ====================

# Core components
embeddings_manager: Optional[EmbeddingsManager] = None
vector_store: Optional[VectorStore] = None
semantic_retrieval: Optional[SemanticRetrieval] = None
agent: Optional[PortfolioAgent] = None
github_fetcher: Optional[GitHubDataFetcher] = None
resume_processor: Optional[ResumeProcessor] = None

# Session management
sessions: Dict[str, ContextManager] = {}


# ==================== INITIALIZATION ====================

def initialize_components():
    """Initialize all chatbot components"""
    global embeddings_manager, vector_store, semantic_retrieval, agent
    global github_fetcher, resume_processor
    
    logger.info("🚀 Initializing chatbot components...")
    
    try:
        # 1. Initialize embeddings manager
        embeddings_manager = EmbeddingsManager(model=EMBEDDING_MODEL)
        logger.info("✅ Embeddings manager initialized")
        
        # 2. Initialize vector store
        vector_store = VectorStore(embedding_dim=1536)
        logger.info("✅ Vector store initialized")
        
        # 3. Initialize semantic retrieval
        semantic_retrieval = SemanticRetrieval(vector_store, embeddings_manager)
        logger.info("✅ Semantic retrieval initialized")
        
        # 4. Initialize GitHub fetcher
        github_fetcher = GitHubDataFetcher(GITHUB_USERNAME, token=GITHUB_TOKEN)
        logger.info("✅ GitHub fetcher initialized")
        
        # 5. Initialize resume processor
        if os.path.exists(RESUME_PATH):
            resume_processor = ResumeProcessor(RESUME_PATH)
            logger.info("✅ Resume processor initialized")
        else:
            logger.warning(f"⚠️  Resume not found at {RESUME_PATH}")
        
        # 6. Index knowledge base
        index_knowledge_base()
        
        # 7. Initialize agent
        if openai_client:
            context_manager = ContextManager(
                window_size=MAX_CONVERSATION_TURNS,
                max_tokens=MAX_CONTEXT_TOKENS,
                openai_client=openai_client
            )
            
            agent = PortfolioAgent(
                openai_client=openai_client,
                context_manager=context_manager,
                model=OPENAI_MODEL,
                max_reasoning_steps=MAX_REASONING_STEPS
            )
            
            # Register tools
            agent.register_tool(AgentTool(
                name="search_resume",
                description="Search resume for skills and experience",
                func=search_resume_knowledge
            ))
            
            agent.register_tool(AgentTool(
                name="search_github",
                description="Search GitHub projects and repositories",
                func=search_github_knowledge
            ))
            
            logger.info("✅ Agent initialized with tools")
        else:
            logger.warning("⚠️  OpenAI client not available. Agent disabled.")
        
        logger.info("✅ All components initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        return False


def index_knowledge_base():
    """Index resume and GitHub data into vector store"""
    try:
        indexer = DocumentIndexer(
            vector_store=vector_store,
            embeddings_manager=embeddings_manager,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        documents_to_index = []
        
        # Index resume
        if resume_processor:
            resume_text = resume_processor.format_resume_for_rag()
            if resume_text:
                documents_to_index.append({
                    "id": "resume",
                    "text": resume_text,
                    "source": "resume",
                    "metadata": {"file": RESUME_PATH}
                })
        
        # Index GitHub profile
        if github_fetcher:
            github_text = github_fetcher.format_profile_for_rag()
            if github_text:
                documents_to_index.append({
                    "id": "github",
                    "text": github_text,
                    "source": "github",
                    "metadata": {"username": GITHUB_USERNAME}
                })
        
        # Index portfolio data
        if os.path.exists(PORTFOLIO_DATA_PATH):
            with open(PORTFOLIO_DATA_PATH, 'r') as f:
                portfolio_data = json.load(f)
                portfolio_text = json.dumps(portfolio_data, indent=2)
                documents_to_index.append({
                    "id": "portfolio",
                    "text": portfolio_text,
                    "source": "portfolio",
                    "metadata": {"type": "portfolio"}
                })
        
        # Index all documents
        if documents_to_index:
            total_chunks = indexer.index_multiple_documents(documents_to_index)
            logger.info(f"✅ Knowledge base indexed: {total_chunks} chunks from {len(documents_to_index)} sources")
        else:
            logger.warning("⚠️  No documents to index")
            
    except Exception as e:
        logger.error(f"❌ Knowledge base indexing error: {e}")


def search_resume_knowledge(query: str) -> str:
    """Tool function: search resume"""
    try:
        results = semantic_retrieval.retrieve_by_source(
            query=query,
            source_type="resume",
            top_k=3
        )
        
        if results:
            return "\n".join([f"- {r['text']}" for r in results])
        return "No matching resume information found."
    except Exception as e:
        return f"Error searching resume: {e}"


def search_github_knowledge(query: str) -> str:
    """Tool function: search GitHub"""
    try:
        results = semantic_retrieval.retrieve_by_source(
            query=query,
            source_type="github",
            top_k=3
        )
        
        if results:
            return "\n".join([f"- {r['text']}" for r in results])
        return "No matching GitHub information found."
    except Exception as e:
        return f"Error searching GitHub: {e}"


def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Get or create a conversation session"""
    if session_id and session_id in sessions:
        return session_id
    
    # Create new session
    new_session_id = f"session_{datetime.now().timestamp()}"
    sessions[new_session_id] = ContextManager(
        window_size=MAX_CONVERSATION_TURNS,
        max_tokens=MAX_CONTEXT_TOKENS,
        openai_client=openai_client
    )
    
    # Set external context
    if resume_processor:
        sessions[new_session_id].set_external_context(
            "resume",
            resume_processor.format_resume_for_rag()
        )
    
    if github_fetcher:
        sessions[new_session_id].set_external_context(
            "github",
            github_fetcher.format_profile_for_rag()
        )
    
    return new_session_id


# ==================== API ENDPOINTS ====================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("🚀 Chatbot starting up...")
    initialize_components()
    logger.info("✅ Chatbot ready!")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        initialized=agent is not None,
        components={
            "embeddings": embeddings_manager is not None,
            "vector_store": vector_store is not None,
            "agent": agent is not None,
            "github_fetcher": github_fetcher is not None,
            "resume_processor": resume_processor is not None,
            "vector_store_size": vector_store.size() if vector_store else 0,
        }
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        # Validate
        if not request.query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if not openai_client or not agent:
            raise HTTPException(
                status_code=503,
                detail="Chatbot not fully initialized. Check OpenAI API key."
            )
        
        # Get or create session
        session_id = get_or_create_session(request.session_id)
        session_context = sessions[session_id]
        
        # Use agent reasoning loop
        result = await agent.reason_and_respond(
            user_query=request.query,
            include_reasoning=request.include_reasoning
        )
        
        # Add to session memory
        session_context.add_conversation_turn(
            user_message=request.query,
            assistant_response=result.get("response", ""),
            confidence=result.get("confidence", 0.0),
            sources=result.get("sources", []),
            metadata={"steps": result.get("steps", 0)}
        )
        
        return ChatResponse(
            response=result.get("response", ""),
            session_id=session_id,
            sources=result.get("sources", []),
            confidence=result.get("confidence", 0.0),
            reasoning=result.get("reasoning"),
            metadata={
                "steps": result.get("steps", 0),
                "context_summary": session_context.get_context_summary()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")


@app.get("/search")
async def search(query: str, source: Optional[str] = None, top_k: int = 5):
    """Search knowledge base"""
    try:
        if not semantic_retrieval:
            raise HTTPException(status_code=503, detail="Search not available")
        
        if source:
            results = semantic_retrieval.retrieve_by_source(query, source, top_k)
        else:
            results = semantic_retrieval.retrieve(query, top_k=top_k)
        
        return {
            "query": query,
            "source": source,
            "results": results,
            "count": len(results)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get session information"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    return {
        "session_id": session_id,
        "context": session.get_context_summary(),
        "conversation_turns": len(session.sliding_window.conversation_history),
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "deleted", "session_id": session_id}
    
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/github/profile")
async def get_github_profile():
    """Get GitHub profile information"""
    if not github_fetcher:
        raise HTTPException(status_code=503, detail="GitHub fetcher not available")
    
    try:
        stats = github_fetcher.fetch_user_stats()
        return stats
    except Exception as e:
        logger.error(f"❌ GitHub profile error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/resume/info")
async def get_resume_info():
    """Get resume information"""
    if not resume_processor:
        raise HTTPException(status_code=503, detail="Resume processor not available")
    
    try:
        resume_data = resume_processor.process_resume()
        return {
            "sections": resume_data.get("sections", {}),
            "contact": resume_data.get("contact", {}),
            "skills": resume_data.get("skills", []),
        }
    except Exception as e:
        logger.error(f"❌ Resume info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get chatbot statistics"""
    return {
        "sessions_active": len(sessions),
        "vector_store_size": vector_store.size() if vector_store else 0,
        "total_conversation_turns": sum(
            len(s.sliding_window.conversation_history)
            for s in sessions.values()
        ),
        "timestamp": datetime.now().isoformat(),
    }


# ==================== ROOT ENDPOINT ====================

@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "name": "🎯 Advanced AI Portfolio Chatbot",
        "version": "3.0.0",
        "status": "running",
        "documentation": "/docs",
        "endpoints": {
            "chat": "POST /chat - Send a chat message",
            "search": "GET /search - Search knowledge base",
            "health": "GET /health - Health check",
            "session": "GET /session/{id} - Session info",
            "github": "GET /github/profile - GitHub profile",
            "resume": "GET /resume/info - Resume information",
            "stats": "GET /stats - Chatbot statistics",
        }
    }


# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"🚀 Starting server on {host}:{port}")
    
    uvicorn.run(
        "main_production:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
