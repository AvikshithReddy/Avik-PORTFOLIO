# 🤖 AI-Powered Portfolio Chatbot

An intelligent chatbot that demonstrates **RAG (Retrieval Augmented Generation)**, **LLM integration**, and **vector databases**—showcasing your ML/AI expertise directly on your portfolio.

## ✨ Features

### 🧠 Advanced ML/AI Techniques
- **RAG**: Retrieves relevant portfolio data and embeds using OpenAI
- **Vector Embeddings**: Uses pgvector for semantic similarity search
- **LLM**: Powered by GPT-4 Turbo for intelligent responses
- **Memory**: Maintains conversation context per session
- **Agent-based**: Autonomous response generation with document grounding

### 🎨 Beautiful UI
- Sleek chat widget integrated into your portfolio
- Responsive design (works on mobile)
- Smooth animations and transitions
- Shows sources for transparency
- Conversation memory visualization

### 🚀 Production-Ready
- Docker containerized (PostgreSQL + FastAPI)
- REST API with complete documentation
- Proper error handling and logging
- CORS support for secure cross-origin requests
- Session management

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose (you have PostgreSQL running ✅)
- OpenAI API Key (get from https://platform.openai.com/api-keys)
- Your resume (PDF) - avikshithReddy_resume.pdf ✅

### 1️⃣ Configure Environment

```bash
cd /Users/avikshithreddy/Desktop/Portfolio/Avik-PORTFOLIO
cp chatbot-backend/.env.example chatbot-backend/.env
```

Edit `chatbot-backend/.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-key-here
```

### 2️⃣ Run Setup Script

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- ✅ Start PostgreSQL with pgvector
- ✅ Build and run FastAPI backend
- ✅ Process and embed your resume
- ✅ Initialize the vector database

### 3️⃣ Open Your Portfolio

```
file:///Users/avikshithreddy/Desktop/Portfolio/Avik-PORTFOLIO/index.html
```

### 4️⃣ Test the Chatbot

Click the chat bubble (bottom right) and ask:
- "Tell me about yourself"
- "What are your key skills?"
- "Describe your ML experience"

---

## 📊 Architecture

```
Your Portfolio (HTML)
        ↓
   chatbot.js (Widget)
        ↓
  FastAPI Backend (RAG Agent)
        ↓
    ┌────────────────────┐
    │  OpenAI GPT-4      │ (LLM)
    │  + Embeddings      │
    └────────────────────┘
    ┌────────────────────┐
    │  PostgreSQL        │
    │  + pgvector        │ (Vector DB)
    └────────────────────┘
```

### How It Works
1. Your **resume is parsed and chunked**
2. Chunks are **embedded using OpenAI's API**
3. **Embeddings stored in PostgreSQL** with pgvector
4. User query is **embedded and searched** against stored docs
5. **Top matching documents become context** for LLM
6. **GPT-4 generates response** using context
7. Response includes **source attribution**
8. **Conversation history maintained** per session

---

## 📁 Project Structure

```
Avik-PORTFOLIO/
├── index.html                    # Your portfolio (updated)
├── chatbot.js                    # Frontend widget
├── portfolio_data.json           # Portfolio metadata
├── avikshithReddy_resume.pdf     # Your resume
│
├── docker-compose.yml            # Docker services
├── chatbot-backend/
│   ├── main.py                   # FastAPI app
│   ├── rag_agent.py              # RAG logic
│   ├── document_processor.py     # Document embedding
│   ├── database.py               # SQLAlchemy models
│   ├── config.py                 # Configuration
│   ├── requirements.txt          # Dependencies
│   └── Dockerfile                # Container image
│
├── CHATBOT_SETUP.md              # Detailed guide
├── DEPLOYMENT.md                 # Deployment instructions
└── setup.sh / setup.bat          # Quick setup
```

---

## 🔌 API Endpoints

```bash
# Chat with the AI
POST http://localhost:8000/api/chat
{
  "query": "Tell me about your experience",
  "session_id": "optional-session"
}

# Health check
GET http://localhost:8000/health

# Initialize documents
POST http://localhost:8000/api/init

# Swagger API docs
http://localhost:8000/docs
```

---

## 🛠️ Docker Commands

```bash
# View services
docker-compose ps

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down

# Rebuild
docker-compose build --no-cache
```

---

## 🎓 What This Demonstrates

✅ **ML/AI Skills**:
- Embedding models & semantic search
- RAG (Retrieval Augmented Generation)
- LLM integration & prompt engineering
- Vector database operations

✅ **Backend Skills**:
- FastAPI & REST API design
- SQLAlchemy & database models
- pgvector integration
- Configuration management

✅ **DevOps Skills**:
- Docker containerization
- Docker Compose orchestration
- Environment management
- Service orchestration

✅ **Frontend Skills**:
- JavaScript widget development
- Real-time API communication
- Session management
- Responsive UI design

---

## 📚 Documentation

- **Setup Guide**: See [CHATBOT_SETUP.md](CHATBOT_SETUP.md)
- **Deployment**: See [DEPLOYMENT.md](DEPLOYMENT.md)
- **API Docs**: http://localhost:8000/docs

---

## 🐛 Quick Troubleshooting

```bash
# Check if services are running
curl http://localhost:8000/health

# View backend logs
docker-compose logs backend

# Check database
docker exec -it portfolio-postgres psql -U portfolio_user -d portfolio_ai

# Reinitialize documents
curl -X POST http://localhost:8000/api/init
```

---

## 🚀 Production Deployment

Set production environment:
```bash
OPENAI_API_KEY=sk-prod-key
DATABASE_URL=postgresql://prod-user:pass@prod-db/db
DEBUG=False
CORS_ORIGINS=["https://yourdomain.com"]
```

Deploy to cloud: AWS ECS, Heroku, Railway, GCP Cloud Run, etc.

---

**Built with ❤️ to showcase your ML/AI expertise**

This demonstrates enterprise-grade RAG, LLM integration, and DevOps capabilities! 🚀
