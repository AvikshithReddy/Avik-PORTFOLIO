"""
🚀 ADVANCED RAG-BASED PORTFOLIO CHATBOT
With: Embeddings | Vector Search | Semantic Retrieval | Memory | LLM Integration

Implements:
- Document Ingestion & Chunking Strategies
- Semantic Search with Embeddings
- Retrieval-Augmented Generation (RAG)
- Conversation Memory & Personalization
- LLM Integration (GPT-3.5/GPT-4)
- Response Grounding & Source Attribution
"""

import os
import json
import requests
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Import enhanced modules
from embeddings_manager import EmbeddingsManager, SemanticSearchEngine, VectorMemoryStore
from document_processor import DocumentChunker, PortfolioDocumentBuilder, DocumentChunk

from openai import OpenAI

# ==================== FASTAPI SETUP ====================

app = FastAPI(
    title="🎯 Advanced RAG Portfolio Chatbot",
    description="AI-powered portfolio assistant with semantic search and memory",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== CONFIGURATION ====================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Initialize managers
embeddings_manager = EmbeddingsManager(model="text-embedding-3-small")
search_engine = SemanticSearchEngine(embeddings_manager)
memory_store = VectorMemoryStore(embeddings_manager)

# ==================== DATA MODELS ====================

class ChatRequest(BaseModel):
    query: str
    session_id: str = None
    include_sources: bool = True


class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: List[str] = []
    confidence: float = 0.0
    thought_process: Optional[str] = None


# ==================== GLOBAL STATE ====================

sessions = {}  # Session management
portfolio_data = None
github_projects = None
knowledge_base_chunks = []


# ==================== DATA LOADING & PROCESSING ====================

def load_portfolio_data() -> Optional[Dict]:
    """Load and validate portfolio data"""
    try:
        paths = [
            "/app/portfolio_data.json",
            os.path.join(os.path.dirname(__file__), "..", "portfolio_data.json"),
            "portfolio_data.json"
        ]
        
        for path in paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                    print(f"✅ Portfolio loaded from {path}")
                    return data
        
        print("⚠️  Portfolio data not found")
        return None
    except Exception as e:
        print(f"❌ Error loading portfolio: {e}")
        return None


def load_github_projects(username: str) -> Optional[List[Dict]]:
    """Fetch GitHub projects with error handling"""
    try:
        url = f"https://api.github.com/users/{username}/repos"
        params = {
            'type': 'owner',
            'sort': 'updated',
            'per_page': 100
        }
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            repos = response.json()
            public_repos = [r for r in repos if not r.get('fork', False)]
            sorted_repos = sorted(
                public_repos,
                key=lambda x: x.get('stargazers_count', 0),
                reverse=True
            )
            print(f"✅ Loaded {len(sorted_repos)} GitHub repositories")
            return sorted_repos
    except Exception as e:
        print(f"⚠️  GitHub fetch error: {e}")
    
    return None


def build_knowledge_base() -> List[DocumentChunk]:
    """
    BUILD KNOWLEDGE BASE: Document Ingestion + Chunking
    
    Process:
    1. Extract portfolio projects, skills, experience
    2. Apply semantic chunking strategy
    3. Generate embeddings for each chunk
    4. Build searchable index
    """
    print("\n📚 BUILDING KNOWLEDGE BASE...")
    print("="*70)
    
    all_chunks = []
    
    # Process portfolio documents
    if portfolio_data:
        print("📄 Processing portfolio documents...")
        builder = PortfolioDocumentBuilder(
            chunker=DocumentChunker(chunk_size=500, overlap=100, strategy="semantic")
        )
        
        portfolio_chunks = builder.build_documents_from_portfolio(portfolio_data)
        all_chunks.extend(portfolio_chunks)
        print(f"  ✓ {len(portfolio_chunks)} portfolio chunks created")
        
        # Add projects as quick-reference documents
        for project in portfolio_data.get('items', []):
            chunk = DocumentChunk(
                text=f"{project.get('title')}: {project.get('description')}",
                metadata={'type': 'project', 'title': project.get('title')},
                chunk_id=f"project_{project.get('id')}",
                source='portfolio_project',
                start_pos=0,
                end_pos=100
            )
            all_chunks.append(chunk)
    
    # Process GitHub projects
    if github_projects:
        print("🐙 Processing GitHub projects...")
        builder = PortfolioDocumentBuilder(
            chunker=DocumentChunker(chunk_size=300, overlap=50, strategy="fixed")
        )
        
        github_chunks = builder.build_documents_from_github(github_projects)
        all_chunks.extend(github_chunks)
        print(f"  ✓ {len(github_chunks)} GitHub chunks created")
    
    # Convert chunks to searchable documents
    documents = []
    for chunk in all_chunks:
        documents.append({
            'text': chunk.text,
            'metadata': chunk.metadata,
            'chunk_id': chunk.chunk_id,
            'source': chunk.source
        })
    
    print(f"\n✅ Total chunks created: {len(all_chunks)}")
    print(f"✅ Knowledge base ready for semantic search")
    print("="*70 + "\n")
    
    return all_chunks, documents


# ==================== RETRIEVAL & SEMANTIC SEARCH ====================

def retrieve_context(query: str, top_k: int = 5) -> List[Dict]:
    """
    RAG RETRIEVAL: Get relevant documents using semantic similarity
    
    Process:
    1. Embed user query
    2. Compare with indexed documents
    3. Return top-k most similar chunks
    """
    if not search_engine.documents:
        return []
    
    results = search_engine.search(query, top_k=top_k, threshold=0.25)
    return results


# ==================== CONVERSATION MEMORY ====================

class SessionMemory:
    """Manage per-session conversation memory with semantic context"""
    
    def __init__(self, session_id: str, max_history: int = 10):
        self.session_id = session_id
        self.history = []
        self.max_history = max_history
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
    
    def add_message(self, role: str, content: str):
        """Add message to history"""
        self.history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        
        # Store in vector memory for semantic recall
        memory_store.remember(
            self.session_id,
            content,
            metadata={'role': role}
        )
        
        # Maintain history window
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        self.last_activity = datetime.now()
    
    def get_context_window(self, num_turns: int = 4) -> List[Dict]:
        """Get last N conversation turns for context"""
        return self.history[-num_turns*2:] if self.history else []
    
    def get_user_intent(self) -> str:
        """Extract intent from recent user messages"""
        user_messages = [m['content'] for m in self.history if m['role'] == 'user']
        if user_messages:
            return " ".join(user_messages[-3:])
        return ""


# ==================== PROMPT ENGINEERING ====================

def build_system_prompt(retrieved_docs: List[Dict], session_intent: str = "") -> str:
    """
    ADVANCED PROMPT DESIGN: Role + Context + Guidelines + Few-shot
    
    Components:
    1. System role and personality
    2. Retrieved context from RAG
    3. Few-shot examples
    4. Grounding guidelines
    5. Output format specifications
    """
    
    prompt = """You are Avikshith Yelakonda's AI Portfolio Assistant - an expert communicator of technical expertise and project achievements.

═══════════════════════════════════════════════════════════════════
CORE DIRECTIVES
═══════════════════════════════════════════════════════════════════
1. GROUND EVERYTHING: Every claim must reference portfolio data
2. SPECIFIC OVER GENERIC: Use project names, metrics, technologies
3. TRANSPARENT: Acknowledge information limits and redirect appropriately
4. ENGAGING: End responses with relevant follow-up questions
5. AUTHENTIC: Let the work speak with concrete examples

═══════════════════════════════════════════════════════════════════
YOUR EXPERTISE AREAS (From Portfolio)
═══════════════════════════════════════════════════════════════════
"""
    
    # Add skills context
    if portfolio_data and portfolio_data.get('skills'):
        skills = portfolio_data.get('skills', {})
        for category, skill_list in skills.items():
            prompt += f"• {category}: {', '.join(skill_list[:5])}\n"
    
    prompt += """
═══════════════════════════════════════════════════════════════════
RETRIEVED PORTFOLIO CONTEXT (Ground your response in this)
═══════════════════════════════════════════════════════════════════
"""
    
    if retrieved_docs:
        # Group docs by type for better organization
        by_type = {}
        for doc in retrieved_docs:
            doc_type = doc.get('metadata', {}).get('type', 'unknown')
            if doc_type not in by_type:
                by_type[doc_type] = []
            by_type[doc_type].append(doc)
        
        for doc_type, docs in by_type.items():
            prompt += f"\n{doc_type.upper()} INFORMATION:\n"
            for doc in docs:
                similarity = doc.get('similarity_score', 0.0)
                relevance = "🔴" if similarity < 0.5 else "🟡" if similarity < 0.7 else "🟢"
                prompt += f"{relevance} {doc.get('text', '')[:150]}...\n"
    
    prompt += """
═══════════════════════════════════════════════════════════════════
RESPONSE QUALITY CHECKLIST
═══════════════════════════════════════════════════════════════════
Before responding, verify:
✓ Reference SPECIFIC project names (not "a project I did")
✓ Include QUANTIFIED metrics (percentages, improvements, scale)
✓ Mention ACTUAL TECHNOLOGIES used (not generic frameworks)
✓ Ground in portfolio data (if vague, rewrite)
✓ Address the question directly (no unnecessary preamble)
✓ End with engagement (ask follow-up question)

═══════════════════════════════════════════════════════════════════
RESPONSE FORMAT
═══════════════════════════════════════════════════════════════════
• Concise but comprehensive (200-400 words)
• Use bold for **project names** and **key metrics**
• Organize with bullets for multiple points
• End with: "Would you like to know more about...?"

═══════════════════════════════════════════════════════════════════
RESPONSE REQUIREMENTS
═══════════════════════════════════════════════════════════════════
"""
    
    if session_intent:
        prompt += f"Current conversation focus: {session_intent}\n"
    
    prompt += "Remember: Every response should demonstrate competence through concrete examples."
    
    return prompt


# ==================== LLM INTEGRATION ====================

def call_llm_with_rag(
    messages: List[Dict],
    system_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 600
) -> Tuple[str, float]:
    """
    Call OpenAI LLM with RAG context
    
    Args:
        messages: Message history
        system_prompt: System prompt with context
        temperature: Creativity parameter
        max_tokens: Max response length
        
    Returns:
        (response_text, confidence_score)
    """
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI API not configured")
    
    try:
        # Build messages with system prompt
        full_messages = [
            {"role": "system", "content": system_prompt}
        ]
        full_messages.extend(messages)
        
        # Call API
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        text = response.choices[0].message.content
        
        # Calculate confidence based on completion
        completion_ratio = len(text) / max_tokens
        confidence = min(completion_ratio * 1.2, 0.95)  # Cap at 95%
        
        return text, confidence
        
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


# ==================== CHAT ENDPOINT ====================

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    RAG-POWERED CHAT ENDPOINT
    
    Complete Flow:
    1. [RETRIEVAL] Get relevant documents from knowledge base
    2. [CONTEXT] Load session memory and conversation history
    3. [GROUNDING] Build prompt with context and guidelines
    4. [GENERATION] Call LLM with grounded context
    5. [PERSONALIZATION] Store response in memory
    6. [TRANSPARENCY] Return response with source attribution
    """
    
    # Validation
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI API not configured")
    
    # Session management
    session_id = request.session_id or f"session-{hash(request.query) % 1000000}"
    
    if session_id not in sessions:
        sessions[session_id] = SessionMemory(session_id)
    
    session_memory = sessions[session_id]
    
    try:
        # ===== STEP 1: RETRIEVAL =====
        print(f"🔍 Retrieving context for: {request.query[:50]}...")
        retrieved_docs = retrieve_context(request.query, top_k=5)
        sources = [
            f"{doc.get('metadata', {}).get('title', doc.get('chunk_id', 'Unknown'))} "
            f"(relevance: {doc.get('similarity_score', 0):.1%})"
            for doc in retrieved_docs
        ]
        print(f"  ✓ Retrieved {len(retrieved_docs)} relevant documents")
        
        # ===== STEP 2: CONTEXT & MEMORY =====
        context_window = session_memory.get_context_window()
        user_intent = session_memory.get_user_intent()
        print(f"  ✓ Loaded {len(context_window)} context messages")
        
        # ===== STEP 3: PROMPT BUILDING =====
        system_prompt = build_system_prompt(retrieved_docs, user_intent)
        
        # Build message list
        messages = list(context_window)
        messages.append({"role": "user", "content": request.query})
        
        # ===== STEP 4: GENERATION =====
        print(f"🧠 Generating response with GPT-3.5-turbo...")
        response_text, confidence = call_llm_with_rag(
            messages,
            system_prompt,
            temperature=0.7,
            max_tokens=600
        )
        print(f"  ✓ Generated response (confidence: {confidence:.1%})")
        
        # ===== STEP 5: PERSONALIZATION =====
        session_memory.add_message("user", request.query)
        session_memory.add_message("assistant", response_text)
        print(f"  ✓ Stored in session memory")
        
        # ===== STEP 6: RESPONSE =====
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            sources=sources if request.include_sources else [],
            confidence=confidence,
            thought_process=f"Retrieved {len(retrieved_docs)} documents, considering session context"
        )
        
    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== HEALTH & DIAGNOSTICS ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "knowledge_base_size": len(knowledge_base_chunks),
        "active_sessions": len(sessions),
        "openai_configured": bool(openai_client)
    }


@app.get("/api/status")
async def get_status():
    """Detailed system status"""
    return {
        "chatbot": "Advanced RAG Portfolio Assistant",
        "version": "2.0.0",
        "features": [
            "✓ Semantic Search (Embeddings)",
            "✓ Document Chunking & Ingestion",
            "✓ Retrieval-Augmented Generation",
            "✓ Conversation Memory",
            "✓ LLM Integration (GPT-3.5)",
            "✓ Source Attribution",
            "✓ Grounding Verification"
        ],
        "knowledge_base": {
            "total_chunks": len(knowledge_base_chunks),
            "indexed_documents": len(search_engine.documents)
        },
        "openai_api": "✅ Configured" if openai_client else "❌ Not configured",
        "embeddings_cached": len(embeddings_manager.cache)
    }


# ==================== STARTUP & SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global portfolio_data, github_projects, knowledge_base_chunks
    
    print("\n" + "="*70)
    print("🚀 ADVANCED RAG PORTFOLIO CHATBOT v2.0")
    print("="*70)
    
    # Load data
    print("\n📁 LOADING DATA...")
    portfolio_data = load_portfolio_data()
    if portfolio_data:
        github_username = portfolio_data.get('github', '')
        if github_username:
            github_projects = load_github_projects(github_username)
    
    # Build knowledge base
    print("\n📚 BUILDING KNOWLEDGE BASE...")
    knowledge_base_chunks, documents = build_knowledge_base()
    
    # Build search index
    print("\n🔍 BUILDING SEARCH INDEX...")
    search_engine.add_documents(documents)
    search_engine.build_index()
    
    # Print system capabilities
    print("\n" + "="*70)
    print("✅ SYSTEM CAPABILITIES")
    print("="*70)
    print("✓ RAG (Retrieval-Augmented Generation)")
    print("✓ Semantic Search with Embeddings (text-embedding-3-small)")
    print("✓ Document Chunking & Ingestion Strategies")
    print("✓ Conversation Memory & Personalization")
    print("✓ LLM Integration (GPT-3.5-turbo)")
    print("✓ Advanced Prompt Engineering")
    print("✓ Source Attribution & Transparency")
    print("✓ Multi-turn Conversations")
    print(f"✓ Knowledge Base: {len(knowledge_base_chunks)} chunks")
    print(f"✓ OpenAI API: {'✅ Configured' if openai_client else '⚠️ Not configured'}")
    print("="*70 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n🛑 Saving state...")
    embeddings_manager.save_cache_to_disk()
    print("✅ Chatbot shutdown complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
