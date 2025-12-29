"""
Enhanced Portfolio Chatbot - Uses Resume, Portfolio & GitHub Data
"""
import os
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai

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
else:
    openai.api_key = openai_api_key

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
        # Try to load from file in the same directory as this script
        portfolio_file = os.path.join(os.path.dirname(__file__), "..", "portfolio_data.json")
        with open(portfolio_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Could not load portfolio data: {e}")
        return None


def load_github_projects(github_username):
    """Fetch projects from GitHub API"""
    try:
        url = f"https://api.github.com/users/{github_username}/repos"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            repos = response.json()
            # Get top repos by stars
            top_repos = sorted(repos, key=lambda x: x.get("stargazers_count", 0), reverse=True)[:5]
            return top_repos
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
    
    # Build the prompt
    prompt = """You are Avikshith Yelakonda's professional AI portfolio assistant.

Your role: Help visitors learn about Avikshith's expertise, projects, and skills. Frame all responses using the actual portfolio data provided below.

====================
PERSONAL INFORMATION
===================="""
    
    if portfolio_data:
        prompt += f"""
Name: {portfolio_data.get('name', 'Avikshith Yelakonda')}
Title: {portfolio_data.get('title', 'ML/AI Professional')}
Bio: {portfolio_data.get('bio', '')}
GitHub: https://github.com/{portfolio_data.get('github', 'avikshithreddy')}
"""
    
        # Add skills
        skills = portfolio_data.get("skills", {})
        if skills:
            prompt += """
====================
SKILLS & EXPERTISE
====================
"""
            if skills.get("core"):
                prompt += f"Core: {', '.join(skills['core'])}\n"
            if skills.get("backend"):
                prompt += f"Backend: {', '.join(skills['backend'])}\n"
            if skills.get("data"):
                prompt += f"Data & ML: {', '.join(skills['data'])}\n"
            if skills.get("tools"):
                prompt += f"Tools & Platforms: {', '.join(skills['tools'])}\n"
        
        # Add projects
        items = portfolio_data.get("items", [])
        if items:
            prompt += """
====================
KEY PROJECTS
====================
"""
            for project in items:
                prompt += f"""
Project: {project.get('title', 'N/A')}
Description: {project.get('description', 'N/A')}
Skills: {', '.join(project.get('skills', []))}
"""
        
        # Add experience
        experience = portfolio_data.get("experience", [])
        if experience:
            prompt += """
====================
EXPERIENCE
====================
"""
            for exp in experience:
                prompt += f"""
{exp.get('role', 'N/A')} at {exp.get('company', 'N/A')}
Duration: {exp.get('duration', 'N/A')}
{exp.get('description', 'N/A')}
"""
    
    # Add GitHub projects
    if github_projects:
        prompt += """
====================
GITHUB PROJECTS (Public)
====================
"""
        for repo in github_projects:
            prompt += f"""
Repository: {repo.get('name', 'N/A')}
URL: {repo.get('html_url', 'N/A')}
Description: {repo.get('description', 'No description')}
Stars: {repo.get('stargazers_count', 0)} | Language: {repo.get('language', 'N/A')}
"""
    
    # Add instructions
    prompt += """
====================
RESPONSE GUIDELINES
====================

When answering questions:

1. REFERENCE ACTUAL DATA: Always reference specific projects, skills, or experience from above.
   ✓ Good: "In the NLP Classification Model project, I achieved 94% accuracy using BERT..."
   ✗ Bad: "I have worked on various projects..."

2. PROJECT-SPECIFIC: When asked about projects, provide concrete details from the portfolio.
   - Mention specific technologies used
   - Highlight quantifiable results (40% improvement, 75% reduction, etc.)
   - Connect to the visitor's potential needs

3. SKILLS CONTEXT: When discussing skills, relate them to actual projects.
   Example: "I've used FastAPI with Docker in the RAG Chatbot System and Data Analytics Dashboard"

4. GITHUB PROJECTS: Reference public repos when relevant to the discussion.

5. TONE: Professional but conversational. Show genuine passion for the work.

6. HONESTY: If asked something not in the portfolio, be honest rather than speculate.
   Example: "That's not covered in my current projects, but here's what I do have..."

7. ENGAGEMENT: Ask follow-up questions to understand visitor needs better.
   Example: "Are you interested in learning more about the ML Pipeline Optimization project?"

Remember: You're demonstrating real skills and achievements, not generic capabilities.
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
    """Chat endpoint using portfolio data + OpenAI"""
    
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if not openai_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API not configured")
    
    # Get or create session
    session_id = request.session_id or f"session-{len(sessions)}-{hash(request.query)}"
    
    # Initialize session if new
    if session_id not in sessions:
        sessions[session_id] = []
    
    # Build message history for this session
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    
    # Add previous messages from this session
    messages.extend(sessions[session_id])
    
    # Add current user message
    messages.append({"role": "user", "content": request.query})
    
    try:
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=600
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
