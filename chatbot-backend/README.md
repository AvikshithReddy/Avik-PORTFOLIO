# Portfolio Chatbot - Complete Setup Guide

## 🎯 Overview

An advanced AI-powered chatbot for your portfolio website that provides grounded responses based on your resume, GitHub repositories, portfolio content, and markdown documentation. Built with FastAPI, OpenAI, and RAG (Retrieval-Augmented Generation).

## 🏗️ Architecture

```
chatbot-backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── schemas.py           # Pydantic models
│   ├── llm/
│   │   ├── __init__.py
│   │   └── openai_client.py # OpenAI API wrapper
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── index.py         # RAG index management
│   │   ├── build_docs.py    # Document ingestion
│   │   └── prompts.py       # System prompts
│   ├── sources/
│   │   ├── __init__.py
│   │   ├── local_files.py   # Portfolio/resume/markdown loading
│   │   └── github.py        # GitHub API integration
│   └── utils/
│       ├── __init__.py
│       ├── text.py          # Text processing utilities
│       └── logging.py       # Logging setup
├── data/
│   ├── portfolio_data.json  # Your portfolio data
│   ├── avikshithReddy_resume.pdf  # Your resume
│   ├── PORTFOLIO_SUMMARY.md
│   └── PROJECT_ANALYSIS.md
├── frontend/
│   ├── chatbot.js           # Chatbot widget
│   └── chatbot_test.html    # Test page
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## 📋 Prerequisites

- Python 3.11+
- OpenAI API key
- GitHub Personal Access Token (optional, for GitHub repo ingestion)
- Docker & Docker Compose (optional, for containerized deployment)

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Navigate to backend directory
cd chatbot-backend

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the `chatbot-backend` directory:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# REQUIRED: OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
CHAT_MODEL=gpt-4-turbo-preview
EMBEDDING_MODEL=text-embedding-3-small

# OPTIONAL: GitHub Integration
GITHUB_TOKEN=ghp_your_github_token_here
GITHUB_USERNAME=avikshithreddy

# Security
ADMIN_INGEST_KEY=your-secret-admin-key
```

**⚠️ IMPORTANT: Where to get API keys**

1. **OpenAI API Key**: 
   - Go to https://platform.openai.com/api-keys
   - Create new secret key
   - Add to `.env` as `OPENAI_API_KEY`

2. **GitHub Token** (optional):
   - Go to https://github.com/settings/tokens
   - Generate new token (classic)
   - Select scopes: `public_repo`, `read:user`
   - Add to `.env` as `GITHUB_TOKEN`

### 3. Prepare Your Data

Place your resume in the `data/` folder:

```bash
# Copy your resume PDF
cp /path/to/your/resume.pdf data/avikshithReddy_resume.pdf
```

The system will use:
- ✅ `data/portfolio_data.json` (sample provided)
- ✅ `data/avikshithReddy_resume.pdf` (add your resume here)
- ✅ `data/*.md` (markdown files for additional context)
- ✅ GitHub repos (if token provided)

### 4. Build RAG Index

Start the backend server:

```bash
cd chatbot-backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Build the RAG index (in another terminal):

```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: your-secret-admin-key" \
  -d '{"sources": ["portfolio", "resume", "markdown", "github"], "force_rebuild": true}'
```

Or use Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/ingest",
    headers={"X-Admin-Key": "your-secret-admin-key"},
    json={"sources": ["portfolio", "resume", "markdown", "github"], "force_rebuild": True}
)
print(response.json())
```

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Chat test
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are Avikshith'\''s main skills?", "include_sources": true}'
```

### 6. Frontend Integration

The chatbot widget is already integrated in `index.html`. To test:

1. Open `frontend/chatbot_test.html` in a browser
2. Click "Test Health Endpoint"
3. Try sample chat questions
4. Check the widget in bottom-right corner

For production, update the backend URL in `index.html`:

```javascript
window.chatbotConfig = {
  backendUrl: 'https://your-backend-url.com',  // Change this
  theme: 'light',
  position: 'bottom-right',
  includeSources: true
};
```

## 🐳 Docker Deployment

### Local Docker

```bash
cd chatbot-backend

# Build and run
docker-compose up --build

# In another terminal, build the index
curl -X POST http://localhost:8000/api/ingest \
  -H "X-Admin-Key: your-secret-admin-key" \
  -d '{"force_rebuild": true}'
```

### Production Deployment

#### Google Cloud Run

```bash
# Build and push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT/chatbot-backend

# Deploy
gcloud run deploy chatbot-backend \
  --image gcr.io/YOUR_PROJECT/chatbot-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=sk-xxx,GITHUB_TOKEN=ghp_xxx
```

#### AWS ECS / Azure Container Apps

Use the provided `Dockerfile` and deploy to your preferred container platform.

## 🔧 API Endpoints

### `GET /health`
Health check endpoint

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "rag_index_loaded": true,
  "total_documents": 150
}
```

### `POST /api/chat`
Chat with the AI assistant

**Request:**
```json
{
  "query": "What projects has Avikshith worked on?",
  "session_id": "optional-session-id",
  "include_sources": true
}
```

**Response:**
```json
{
  "response": "Avikshith has worked on several projects...",
  "session_id": "abc-123",
  "confidence": 0.89,
  "sources": [
    {
      "source_type": "portfolio",
      "source_name": "Project: Customer Churn Prediction",
      "locator": "./data/portfolio_data.json",
      "snippet": "Built an end-to-end machine learning...",
      "relevance_score": 0.92
    }
  ]
}
```

### `POST /api/ingest`
Rebuild RAG index (admin only)

**Headers:**
- `X-Admin-Key`: Your admin key from `.env`

**Request:**
```json
{
  "sources": ["portfolio", "resume", "markdown", "github"],
  "force_rebuild": true
}
```

**Response:**
```json
{
  "status": "success",
  "documents_processed": 15,
  "chunks_created": 150,
  "embeddings_generated": 150,
  "sources_ingested": ["portfolio", "resume", "markdown", "github"],
  "errors": []
}
```

## 🎨 Frontend Widget

The chatbot widget is a standalone JavaScript component that can be added to any HTML page.

### Features
- 💬 Floating chat bubble
- 🎯 Session persistence
- 📚 Source citations
- 🎨 Customizable theme
- 📱 Mobile responsive
- ⚡ Real-time responses

### Configuration Options

```javascript
window.chatbotConfig = {
  backendUrl: 'http://localhost:8000',  // Backend API URL
  theme: 'light',                       // 'light' or 'dark'
  position: 'bottom-right',             // Widget position
  includeSources: true                  // Show source citations
};
```

## 🔒 Security Considerations

1. **API Keys**: Never commit `.env` file to git
2. **CORS**: Configure `CORS_ALLOW_ORIGINS` for production
3. **Admin Key**: Use strong `ADMIN_INGEST_KEY` for ingestion endpoint
4. **Rate Limiting**: Consider adding rate limiting in production
5. **HTTPS**: Always use HTTPS in production

## 📊 Monitoring & Logging

Logs are written to:
- Console (stdout)
- `logs/chatbot.log` (file)

Monitor:
- API response times
- RAG retrieval accuracy
- OpenAI API usage
- Error rates

## 🐛 Troubleshooting

### Backend won't start
- Check Python version (3.11+)
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check `.env` file exists and has valid `OPENAI_API_KEY`

### RAG index not loading
- Run ingestion endpoint: `/api/ingest`
- Check data files exist in `data/` folder
- Verify OpenAI API key has embedding model access

### Chat responses are generic
- Ensure RAG index is built successfully
- Check confidence scores in responses
- Verify source documents contain relevant information
- Try rebuilding index with `force_rebuild: true`

### Frontend can't connect to backend
- Verify backend is running: `curl http://localhost:8000/health`
- Check CORS settings in `.env`
- Update `backendUrl` in chatbot config
- Check browser console for errors

## 📝 Customization

### Adding New Data Sources

Edit `app/rag/build_docs.py` and add a new processing method:

```python
def _process_custom_source(self) -> List[Dict[str, Any]]:
    # Your custom data loading logic
    pass
```

### Modifying System Prompt

Edit `app/rag/prompts.py` to customize the chatbot personality and instructions.

### Changing Models

Update `.env`:
```env
CHAT_MODEL=gpt-4-turbo-preview  # or gpt-3.5-turbo for faster/cheaper
EMBEDDING_MODEL=text-embedding-3-small  # or text-embedding-3-large
```

## 🎓 Best Practices

1. **Regular Index Updates**: Rebuild index when content changes
2. **Monitor Costs**: Track OpenAI API usage
3. **Version Control**: Keep data files in git (except sensitive info)
4. **Testing**: Use `chatbot_test.html` to verify functionality
5. **Backup**: Backup RAG index periodically

## 📚 Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [GitHub API Documentation](https://docs.github.com/en/rest)

## 🤝 Support

For issues or questions:
1. Check logs in `logs/chatbot.log`
2. Test with `chatbot_test.html`
3. Verify API keys are valid
4. Check backend status: `/health` endpoint

## 📄 License

MIT License - Feel free to modify and use for your portfolio!

---

**Built with ❤️ using OpenAI, FastAPI, and modern RAG techniques**
