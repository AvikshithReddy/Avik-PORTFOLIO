"""
Enhanced Portfolio Chatbot - Optimized RAG System
Features:
- Persistent embeddings (no re-computation)
- Fast numpy-based vector search
- Smart GitHub caching
- Sub-second response times
"""
import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Import our optimized RAG modules
from rag_engine import RAGEngine
from github_ingestion import GitHubIngestion
from document_processor import DocumentProcessor

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Portfolio AI Chatbot - Optimized RAG")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY", "")
if not openai_api_key:
    print("⚠️  WARNING: OPENAI_API_KEY not set!")
    client = None
    rag_engine = None
else:
    client = OpenAI(api_key=openai_api_key)
    # Initialize RAG engine with persistent embeddings
    rag_engine = RAGEngine(openai_api_key, cache_dir="./embeddings_cache")

# Initialize GitHub ingestion with caching
github_ingestion = GitHubIngestion(cache_dir="./github_cache", cache_hours=24)

# In-memory session storage
sessions = {}

# Request/Response models
class ChatRequest(BaseModel):
    query: str
    session_id: str = None

class ChatResponse(BaseModel):
    response: str
    session_id: str


def load_portfolio_data():
    """Load portfolio data from JSON file"""
    try:
        # Try multiple paths: Docker and local
        paths_to_try = [
            "/app/portfolio_data.json",  # Docker path
            os.path.join(os.path.dirname(__file__), "..", "portfolio_data.json"),  # Local path
            "portfolio_data.json"  # Current directory
        ]
        
        for portfolio_file in paths_to_try:
            if os.path.exists(portfolio_file):
                with open(portfolio_file, "r") as f:
                    return json.load(f)
        
        print(f"⚠️  Could not find portfolio_data.json in any location")
        return None
    except Exception as e:
        print(f"⚠️  Could not load portfolio data: {e}")
        return None


def build_system_prompt():
    """Build comprehensive system prompt with portfolio and GitHub data"""
    
    portfolio_data = load_portfolio_data()
    
    # Build comprehensive system prompt
    prompt = """You are Avikshith Yelakonda's professional AI portfolio assistant.

YOUR PRIMARY ROLE:
- Answer questions about Avikshith's projects, skills, experience, and capabilities
- Ground ALL responses in actual portfolio data provided below
- Help visitors understand what Avikshith can build and deliver
- Demonstrate technical expertise through specific project examples

CRITICAL: ALWAYS reference specific projects, metrics, and technologies when responding.
Never give generic answers. Use the detailed portfolio data to provide concrete examples."""
    
    if portfolio_data:
        prompt += f"""

╔════════════════════════════════════════════════════════════════════╗
║                     PROFESSIONAL PROFILE
╚════════════════════════════════════════════════════════════════════╝

Name: {portfolio_data.get('name', 'Avikshith Yelakonda')}
Title: {portfolio_data.get('title', 'ML/AI Professional | Data Scientist')}
Bio: {portfolio_data.get('bio', '')}
GitHub: https://github.com/{portfolio_data.get('github', 'avikshithreddy')}
"""
    
        # Detailed skills with context
        skills = portfolio_data.get("skills", {})
        if skills:
            prompt += """
╔════════════════════════════════════════════════════════════════════╗
║              TECHNICAL SKILLS & EXPERTISE AREAS
╚════════════════════════════════════════════════════════════════════╝

CORE ML/AI COMPETENCIES (Primary Focus Areas):
"""
            if skills.get("core"):
                prompt += f"  • {', '.join(skills['core'])}\n"
            
            prompt += "\nBACKEND & INFRASTRUCTURE (Production-Ready):\n"
            if skills.get("backend"):
                prompt += f"  • {', '.join(skills['backend'])}\n"
            
            prompt += "\nDATA & ML FRAMEWORKS (Implementation):\n"
            if skills.get("data"):
                prompt += f"  • {', '.join(skills['data'])}\n"
            
            prompt += "\nDEVOPS & TOOLS (Deployment & Workflow):\n"
            if skills.get("tools"):
                prompt += f"  • {', '.join(skills['tools'])}\n"
        
        # Detailed project descriptions with quantifiable impact
        projects = portfolio_data.get("projects", [])
        if projects:
            prompt += """
╔════════════════════════════════════════════════════════════════════╗
║                    FEATURED PROJECTS & ACHIEVEMENTS
╚════════════════════════════════════════════════════════════════════╝
"""
            for i, project in enumerate(projects, 1):
                name = project.get('name', 'N/A')
                description = project.get('description', 'N/A')
                tech = project.get('technologies', [])
                results = project.get('results', 'N/A')
                
                prompt += f"""
PROJECT {i}: {name}
──────────────────────────────────────────────────────────────────────
Description: {description}
Technologies: {', '.join(tech)}
Impact: {results}
"""
        
        # Experience details - NEW STRUCTURE
        experience = portfolio_data.get("experience", [])
        if experience:
            prompt += """
╔════════════════════════════════════════════════════════════════════╗
║                       PROFESSIONAL EXPERIENCE
╚════════════════════════════════════════════════════════════════════╝
"""
            for exp in experience:
                title = exp.get('title', 'N/A')
                company = exp.get('company', 'N/A')
                duration = exp.get('duration', 'N/A')
                location = exp.get('location', 'N/A')
                description = exp.get('description', 'N/A')
                skills = exp.get('skills', [])
                
                prompt += f"""
📌 {title} | {company}
   Duration: {duration} | Location: {location}
   
   Responsibilities & Achievements:
   {description}
   
   Key Skills: {', '.join(skills) if skills else 'N/A'}
   ──────────────────────────────────────────────────────────────────────
"""
    
    # Enhanced response guidelines
    prompt += """
╔════════════════════════════════════════════════════════════════════╗
║                  HOW TO ANSWER QUESTIONS
╚════════════════════════════════════════════════════════════════════╝

WHEN ASKED ABOUT PROJECTS:
├─ Reference the specific project name (e.g., "NLP Classification Model")
├─ Mention quantifiable results (e.g., "94% accuracy", "40% improvement")
├─ List the actual technologies used (BERT, PyTorch, FastAPI, Docker, etc.)
├─ Explain the problem solved and business/technical impact
└─ Example: "In my NLP Classification Model, I used BERT transformers to achieve 
             94% accuracy on multi-label classification, deployed via FastAPI + Docker"

WHEN ASKED ABOUT SKILLS:
├─ Ground in actual project examples
├─ Show depth: "I've used Python for ML pipeline optimization, backend APIs with FastAPI..."
├─ Connect skills to business outcomes
└─ Example: "I use PyTorch and scikit-learn for ML work - most recently in the ML Pipeline 
             Optimization project where I achieved 40% speed improvement"

WHEN ASKED ABOUT CAPABILITIES:
├─ Reference relevant projects that demonstrate the capability
├─ Highlight relevant skills from the list above
├─ Be specific about technologies and methodologies
└─ Example: "I build production ML systems - demonstrated in the RAG Chatbot System 
             with FastAPI, pgvector, and OpenAI integration"

WHEN ASKED ABOUT GITHUB:
├─ Use the RELEVANT CONTEXT provided to answer
├─ Reference specific repository names and details
├─ Connect GitHub work to portfolio project themes
└─ Example: "I have several Python projects on GitHub including data analysis 
             and machine learning work"

TONE & STYLE GUIDELINES:
├─ Professional yet approachable
├─ Confident but not arrogant (let the work speak)
├─ Conversational but structured
├─ Use "I" statements to show ownership
└─ End with engagement questions: "Would you like to know more about...?"

IF INFORMATION NOT IN PORTFOLIO OR CONTEXT:
├─ Be honest: "That's not in my current portfolio, but here's what I do have..."
├─ Pivot to relevant strengths
├─ Offer to discuss related capabilities

QUALITY ASSURANCE CHECKLIST:
✓ Does the answer reference a specific project? If not, add one.
✓ Does it mention actual technologies used? If not, add them from the skills list.
✓ Does it include quantifiable metrics? If not, add impact numbers from descriptions.
✓ Is it grounded in portfolio data? If generic, rewrite with specific examples.
✓ Does it end with engagement? Consider asking a follow-up question.

REMEMBER: You're not a generic AI assistant. You're Avikshith's personal portfolio AI,
demonstrating real skills through real projects. Every answer must be grounded in the
portfolio data and relevant context. Vague or generic answers are NOT acceptable.
"""
    
    return prompt


# Load system prompt at startup (will be reloaded with RAG context per query)
SYSTEM_PROMPT = build_system_prompt()


def initialize_rag_system():
    """Initialize RAG system with all data at startup"""
    if not rag_engine:
        return
    
    print("🔧 Initializing RAG system...")
    
    # Load portfolio data
    portfolio_data = load_portfolio_data()
    if not portfolio_data:
        print("⚠️  No portfolio data found")
        return
    
    # Process portfolio into documents
    portfolio_docs = DocumentProcessor.process_portfolio(portfolio_data)
    
    # Fetch GitHub data with caching
    github_username = portfolio_data.get("github", "avikshithreddy")
    github_repos = github_ingestion.fetch_repositories(
        github_username,
        use_cache=True,
        max_repos=100
    )
    
    # Create GitHub document chunks
    github_docs = github_ingestion.create_document_chunks(github_repos)
    
    # Combine all documents
    all_documents = portfolio_docs + github_docs
    
    print(f"📊 Total documents: {len(all_documents)}")
    print(f"   - Portfolio: {len(portfolio_docs)}")
    print(f"   - GitHub: {len(github_docs)}")
    
    # Add to RAG engine (will use cache if available)
    rag_engine.add_documents(all_documents)
    
    print("✅ RAG system ready!")


@app.on_event("startup")
async def startup():
    """Startup initialization"""
    print("🚀 Portfolio AI Chatbot Starting...")
    print("📍 Optimized RAG version with persistent embeddings")
    print(f"🔑 OpenAI API Key: {'✅ Configured' if openai_api_key else '❌ NOT SET'}")
    print(f"🧠 Model: gpt-3.5-turbo")
    print(f"📊 System Prompt Size: {len(SYSTEM_PROMPT)} characters")
    
    # Initialize RAG system
    initialize_rag_system()
    
    print("✅ Ready to answer questions about Avikshith's portfolio!")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "rag_enabled": rag_engine is not None}


def should_include_resume(query):
    """Detect if query is about skills, resume, or credentials"""
    resume_keywords = [
        "skill", "resume", "cv", "credential", "qualification", 
        "experience", "education", "certificate", "certified",
        "expertise", "proficiency", "competency", "background",
        "pdf", "download", "full resume", "detailed"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in resume_keywords)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint using optimized RAG with fast vector search"""
    
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if not openai_api_key or not client or not rag_engine:
        raise HTTPException(status_code=500, detail="OpenAI API or RAG engine not configured")
    
    try:
        # Get or create session
        session_id = request.session_id or f"session-{len(sessions)}-{hash(request.query)}"
        
        # Initialize session if new
        if session_id not in sessions:
            sessions[session_id] = []
        
        # Fast RAG retrieval using pre-computed embeddings
        print(f"🔍 Query: {request.query}")
        retrieved_docs = rag_engine.retrieve_top_k(
            query=request.query,
            top_k=8,  # Get top 8 most relevant chunks
            min_similarity=0.25
        )
        
        # Format context from retrieved documents
        relevant_context = rag_engine.format_context(retrieved_docs)
        
        print(f"📊 Retrieved {len(retrieved_docs)} relevant documents")
        
        # Check if user is asking about resume/skills
        is_resume_query = should_include_resume(request.query)
        
        # Build enhanced system prompt with retrieved context
        enhanced_system_prompt = SYSTEM_PROMPT
        
        if relevant_context:
            enhanced_system_prompt += f"\n\n{relevant_context}"
        
        # Add resume info conditionally
        if is_resume_query:
            enhanced_system_prompt += """

╔════════════════════════════════════════════════════════════════════╗
║                        RESUME & CREDENTIALS
╚════════════════════════════════════════════════════════════════════╝

For detailed resume information, certifications, and complete background:
📄 Full Resume: https://avikshithreddy.github.io/Avik-PORTFOLIO/avikshithReddy_resume.pdf

When answering about skills/credentials:
- Reference the portfolio skills sections above
- Suggest viewing the full resume for comprehensive details
- Mention: "For a detailed breakdown, see my full resume at [link]"
- Include the resume link in your response
"""
        
        # Build message history for this session
        messages = [
            {"role": "system", "content": enhanced_system_prompt},
        ]
        
        # Add previous messages from this session (limit to last 4 for context)
        if sessions[session_id]:
            messages.extend(sessions[session_id][-4:])
        
        # Add current user message
        messages.append({"role": "user", "content": request.query})
        
        # Call OpenAI API with optimized parameters
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.6,  # Slightly lower for more consistent, grounded responses
            max_tokens=800,  # Increased for more detailed answers
            top_p=0.95  # Better response quality with focused sampling
        )
        
        assistant_message = response.choices[0].message.content
        
        # Store this exchange in session history
        sessions[session_id].append({"role": "user", "content": request.query})
        sessions[session_id].append({"role": "assistant", "content": assistant_message})
        
        return ChatResponse(
            response=assistant_message,
            session_id=session_id
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
