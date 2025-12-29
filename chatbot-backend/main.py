"""
Enhanced Portfolio Chatbot - RAG System with Embeddings
Uses Resume, Portfolio & GitHub Data with Semantic Search
"""
import os
import json
import requests
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import hashlib

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Portfolio AI Chatbot")

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
else:
    client = OpenAI(api_key=openai_api_key)

# In-memory session storage
sessions = {}
embeddings_cache = {}

# Request/Response models
class ChatRequest(BaseModel):
    query: str
    session_id: str = None

class ChatResponse(BaseModel):
    response: str
    session_id: str


def get_embedding(text):
    """Get embedding for text using OpenAI API"""
    try:
        # Hash for caching
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in embeddings_cache:
            return embeddings_cache[text_hash]
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding
        embeddings_cache[text_hash] = embedding
        return embedding
    except Exception as e:
        print(f"⚠️  Could not get embedding: {e}")
        return None


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    if not a or not b:
        return 0
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)


def extract_and_chunk_documents(portfolio_data, github_projects):
    """Extract and chunk documents for semantic search with enhanced detail"""
    documents = []
    
    # Experience chunks - ENHANCED
    if portfolio_data and "experience" in portfolio_data:
        for exp in portfolio_data["experience"]:
            title = exp.get("title", "")
            company = exp.get("company", "")
            duration = exp.get("duration", "")
            description = exp.get("description", "")
            skills = exp.get("skills", [])
            location = exp.get("location", "")
            
            # Main experience chunk
            chunk = f"""CURRENT/RECENT EXPERIENCE: {title} at {company}
Duration: {duration}
Location: {location}
Role Description: {description}
Key Skills: {', '.join(skills)}
"""
            documents.append({"type": "experience", "content": chunk, "metadata": {"title": title, "company": company, "duration": duration, "current": "Present" in duration}})
            
            # Separate chunk for each skill area in this role
            if skills:
                skills_chunk = f"""Skills developed at {company} ({title}): {', '.join(skills)}. These are core competencies Avikshith uses to build solutions."""
                documents.append({"type": "experience_skills", "content": skills_chunk, "metadata": {"company": company}})
    
    # Project chunks - ENHANCED
    if portfolio_data and "projects" in portfolio_data:
        for project in portfolio_data["projects"]:
            name = project.get("name", "")
            description = project.get("description", "")
            tech = project.get("technologies", [])
            results = project.get("results", "")
            
            # Main project chunk
            chunk = f"""PROJECT: {name}
Description: {description}
Technologies Used: {', '.join(tech)}
Results/Impact: {results}
"""
            documents.append({"type": "project", "content": chunk, "metadata": {"name": name, "impact": results}})
            
            # Separate chunk for tech stack discussion
            if tech:
                tech_chunk = f"""In the project '{name}', Avikshith used: {', '.join(tech)}. This demonstrates expertise in {', '.join(tech[:3])}."""
                documents.append({"type": "project_tech", "content": tech_chunk, "metadata": {"project": name}})
    
    # GitHub projects chunks - ENHANCED
    if github_projects:
        for repo in github_projects[:40]:  # Top 40 repos
            name = repo.get("name", "")
            description = repo.get("description", "") or "No description provided"
            language = repo.get("language", "Unknown")
            stars = repo.get("stargazers_count", 0)
            url = repo.get("html_url", "")
            
            chunk = f"""GITHUB REPOSITORY: {name}
Description: {description}
Programming Language: {language}
Stars: {stars}
Repository URL: {url}
"""
            documents.append({"type": "github", "content": chunk, "metadata": {"name": name, "url": url, "stars": stars, "language": language}})
    
    # Skills chunks - ENHANCED
    if portfolio_data and "skills" in portfolio_data:
        skills_obj = portfolio_data["skills"]
        
        # Category-wise chunks
        for category, items in skills_obj.items():
            if items:
                chunk = f"""SKILLS IN {category.upper()}: {', '.join(items)}
Avikshith is proficient in the {category} tools and technologies: {', '.join(items)}"""
                documents.append({"type": "skills", "content": chunk, "metadata": {"category": category, "count": len(items)}})
        
        # Comprehensive skills chunk
        all_skills = []
        for category, items in skills_obj.items():
            all_skills.extend(items)
        if all_skills:
            skills_chunk = f"""COMPLETE TECHNICAL SKILL SET: {', '.join(set(all_skills))}
Avikshith has broad expertise across multiple domains including: {', '.join(set(all_skills))}"""
            documents.append({"type": "all_skills", "content": skills_chunk, "metadata": {"total_skills": len(set(all_skills))}})
    
    # Education chunks
    if portfolio_data and "education" in portfolio_data:
        for edu in portfolio_data["education"]:
            degree = edu.get("degree", "")
            school = edu.get("school", "")
            duration = edu.get("duration", "")
            
            chunk = f"""EDUCATION: {degree} from {school} ({duration})"""
            documents.append({"type": "education", "content": chunk, "metadata": {"school": school, "degree": degree}})
    
    # Profile summary chunk
    if portfolio_data:
        name = portfolio_data.get("name", "Avikshith")
        title = portfolio_data.get("title", "")
        bio = portfolio_data.get("bio", "")
        
        summary_chunk = f"""{name} - {title}
Bio: {bio}
Specialized in building data products with focus on reliability, scalability, and measurable business outcomes."""
        documents.append({"type": "profile", "content": summary_chunk, "metadata": {"name": name}})
    
    print(f"📚 Created {len(documents)} document chunks for semantic search")
    return documents


def retrieve_relevant_context(query, documents, top_k=8):
    """Enhanced retrieval with semantic search and keyword matching"""
    if not documents or not client:
        return ""
    
    try:
        query_embedding = get_embedding(query)
        if not query_embedding:
            return ""
        
        # Score documents with hybrid approach: semantic + keyword
        scored_docs = []
        query_lower = query.lower()
        
        for doc in documents:
            doc_embedding = get_embedding(doc["content"])
            if doc_embedding:
                # Semantic similarity score
                semantic_score = cosine_similarity(query_embedding, doc_embedding)
                
                # Keyword matching boost
                keyword_boost = 0
                if "experience" in query_lower and doc["type"] in ["experience", "experience_skills"]:
                    keyword_boost = 0.3
                elif "project" in query_lower and doc["type"] in ["project", "project_tech"]:
                    keyword_boost = 0.3
                elif "skill" in query_lower and doc["type"] in ["skills", "all_skills"]:
                    keyword_boost = 0.3
                elif "github" in query_lower and doc["type"] == "github":
                    keyword_boost = 0.2
                elif "education" in query_lower and doc["type"] == "education":
                    keyword_boost = 0.3
                
                # Combined score
                combined_score = semantic_score + keyword_boost
                scored_docs.append((combined_score, semantic_score, doc))
        
        # Sort by combined score
        scored_docs.sort(key=lambda x: (x[0], x[1]), reverse=True)
        top_docs = scored_docs[:top_k]
        
        # Build prioritized context
        context = "📋 RELEVANT CONTEXT FROM PORTFOLIO:\n\n"
        added_count = 0
        
        for combined_score, semantic_score, doc in top_docs:
            if semantic_score > 0.25:  # Lower threshold with keyword boost
                doc_type = doc['type'].upper()
                context += f"[{doc_type}]\n{doc['content']}\n"
                added_count += 1
        
        if added_count == 0:
            return ""
        
        print(f"✅ Retrieved {added_count} relevant document(s) for query: {query[:50]}...")
        return context
        
    except Exception as e:
        print(f"⚠️  Error in semantic search: {e}")
        return ""


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


def load_github_projects(github_username):
    """Fetch ALL projects from GitHub API"""
    try:
        # Fetch all repositories (not just top 5)
        url = f"https://api.github.com/users/{github_username}/repos"
        params = {
            'type': 'owner',
            'sort': 'updated',
            'per_page': 100,  # Get up to 100 repos
            'page': 1
        }
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            repos = response.json()
            # Sort by stars and relevance, exclude forks if needed
            public_repos = [r for r in repos if not r.get('fork', False)]
            sorted_repos = sorted(public_repos, key=lambda x: x.get("stargazers_count", 0), reverse=True)
            
            print(f"📊 Found {len(sorted_repos)} public repositories for {github_username}")
            return sorted_repos
    except Exception as e:
        print(f"⚠️  Could not fetch GitHub projects: {e}")
    return None


def build_system_prompt():
    """Build comprehensive system prompt with portfolio and GitHub data"""
    
    portfolio_data = load_portfolio_data()
    github_projects = None
    
    if portfolio_data:
        github_username = portfolio_data.get("github", "")
        if github_username:
            github_projects = load_github_projects(github_username)
    
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
    
    # GitHub projects with specific details
    if github_projects:
        prompt += """
╔════════════════════════════════════════════════════════════════════╗
║              GITHUB PROJECTS (Complete Public Portfolio)
╚════════════════════════════════════════════════════════════════════╝
"""
        # Group by language for better organization
        by_language = {}
        for repo in github_projects:
            lang = repo.get('language', 'Other') or 'Other'
            if lang not in by_language:
                by_language[lang] = []
            by_language[lang].append(repo)
        
        for lang in sorted(by_language.keys()):
            repos_in_lang = by_language[lang]
            prompt += f"\n{lang.upper()} ({len(repos_in_lang)} projects):\n"
            
            for repo in repos_in_lang:
                name = repo.get('name', 'N/A')
                url = repo.get('html_url', 'N/A')
                description = repo.get('description', '')
                stars = repo.get('stargazers_count', 0)
                
                # Build description from repo data
                if description:
                    desc_text = description[:120]
                else:
                    # Infer from repo name if no description
                    desc_text = name.replace('-', ' ').replace('_', ' ')
                
                prompt += f"""
  • {name}
    URL: {url}
    Stars: {stars} | About: {desc_text}
"""
    
    # Enhanced response guidelines with specific instructions
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
├─ Reference ALL public repositories mentioned above
├─ Group by programming language when relevant
├─ Mention specific repository names from the GitHub section
├─ Connect GitHub work to portfolio project themes
├─ For repos with limited descriptions, infer context from name and language
└─ Example: "I have several Python projects on GitHub including Consumer-Affairs---Prediction, 
             A-B-testing-marketing-analysis, and others showcasing my data science work"

WHEN ASKED ABOUT SPECIFIC PROJECT NOT IN PORTFOLIO:
├─ Check GitHub projects section above
├─ Reference the repository if it exists
├─ Provide details from repository name, language, and description
├─ Connect to broader skills and experience
└─ Example: "In my Consumer-Affairs---Prediction project on GitHub, I worked with data analysis 
             and prediction techniques using Python and machine learning"

TONE & STYLE GUIDELINES:
├─ Professional yet approachable
├─ Confident but not arrogant (let the work speak)
├─ Conversational but structured
├─ Use "I" statements to show ownership
└─ End with engagement questions: "Would you like to know more about...?"

IF INFORMATION NOT IN PORTFOLIO:
├─ Be honest: "That's not in my current portfolio, but here's what I do have..."
├─ Pivot to relevant strengths
├─ Offer to discuss related capabilities
└─ Example: "I haven't worked with Kubernetes yet, but I have strong Docker experience 
             and understand containerization principles through the projects I've built"

QUALITY ASSURANCE CHECKLIST:
✓ Does the answer reference a specific project? If not, add one.
✓ Does it mention actual technologies used? If not, add them from the skills list.
✓ Does it include quantifiable metrics? If not, add impact numbers from descriptions.
✓ Is it grounded in portfolio data? If generic, rewrite with specific examples.
✓ Does it end with engagement? Consider asking a follow-up question.

REMEMBER: You're not a generic AI assistant. You're Avikshith's personal portfolio AI,
demonstrating real skills through real projects. Every answer must be grounded in the
portfolio data above. Vague or generic answers are NOT acceptable.
"""
    
    return prompt


# Load system prompt at startup
SYSTEM_PROMPT = build_system_prompt()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint using RAG with semantic search"""
    
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if not openai_api_key or not client:
        raise HTTPException(status_code=500, detail="OpenAI API not configured")
    
    try:
        # Get or create session
        session_id = request.session_id or f"session-{len(sessions)}-{hash(request.query)}"
        
        # Initialize session if new
        if session_id not in sessions:
            sessions[session_id] = []
        
        # Load portfolio data and GitHub projects
        portfolio_data = load_portfolio_data()
        github_projects = load_github_projects("avikshithreddy")
        
        # Extract and chunk documents for RAG
        documents = extract_and_chunk_documents(portfolio_data, github_projects)
        
        # Retrieve relevant context using semantic search - ENHANCED TOP-K
        relevant_context = retrieve_relevant_context(request.query, documents, top_k=8)
        
        # Build enhanced system prompt with retrieved context
        enhanced_system_prompt = SYSTEM_PROMPT
        if relevant_context:
            enhanced_system_prompt += f"\n\n{relevant_context}"
        
        # Build message history for this session
        messages = [
            {"role": "system", "content": enhanced_system_prompt},
        ]
        
        # Add previous messages from this session (limit to last 4 for context)
        if sessions[session_id]:
            messages.extend(sessions[session_id][-4:])
        
        # Add current user message
        messages.append({"role": "user", "content": request.query})
        
        # Call OpenAI API using new client with improved parameters
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
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.on_event("startup")
async def startup():
    """Startup message"""
    print("🚀 Portfolio AI Chatbot Starting...")
    print("📍 Enhanced version - Using Resume, Portfolio & GitHub Data")
    print(f"🔑 OpenAI API Key: {'✅ Configured' if openai_api_key else '❌ NOT SET'}")
    print(f"🧠 Model: gpt-3.5-turbo")
    print(f"📊 System Prompt Size: {len(SYSTEM_PROMPT)} characters")
    print("✅ Ready to answer questions about Avikshith's portfolio!")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
