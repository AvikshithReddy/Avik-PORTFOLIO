# Avik Portfolio – Chatbot + Site (Docker)

Run the FastAPI RAG backend, Qdrant vector DB, and Nginx‑served frontend with one command.

## Stack
- Frontend: static site served by Nginx (port 8080), proxies `/api/*` to backend.
- Backend: FastAPI + LangGraph/LlamaIndex (port 8000), OpenAI + optional Cohere rerank.
- Vector DB: Qdrant (port 6333), data persisted in `chatbot-backend/qdrant_data`.

## Prerequisites
- Docker Desktop (buildx enabled) on Apple Silicon/ARM or x86.
- OpenAI API key; optional Cohere key and GitHub token for repo ingest.

## Quick start
1. Copy env template and fill secrets:
   ```bash
   cd Avik-PORTFOLIO/chatbot-backend
   cp .env.example .env
   # set OPENAI_API_KEY (required), COHERE_API_KEY (optional), GITHUB_TOKEN (optional)
   # set CORS_ALLOW_ORIGINS to your frontend domain (e.g., https://avikshithreddy.github.io)
   ```
2. From repo root, build and start everything:
   ```bash
   cd Avik-PORTFOLIO
   docker compose up -d --build
   ```
3. Open:
   - Site: http://localhost:8080  
   - Backend health: http://localhost:8000/health  
   - Qdrant UI: http://localhost:6333/dashboard

## Ingest data (after containers up)
Populate Qdrant collections with resume/portfolio/GitHub/LinkedIn content:
```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "x-admin-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{"sources":["resume","portfolio","github","linkedin"]}'
```
Force a clean rebuild (clears Qdrant collections first):
```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "x-admin-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{"sources":["resume","portfolio","github","linkedin"],"force_rebuild":true}'
```
If you don’t set `COHERE_API_KEY`, reranking falls back to simple ordering.

### LinkedIn data
Drop LinkedIn exports into `chatbot-backend/data/Profile.pdf` (primary) or `chatbot-backend/data/linkedin/` (optional extra files) before ingesting.

### Refresh local data from source files
Regenerate `portfolio_data.json` from `index.html`, sync resume/LinkedIn PDFs, and optionally trigger ingest:
```bash
python3 chatbot-backend/scripts/refresh_data.py --ingest
```
Watch for changes (live refresh + ingest):
```bash
python3 chatbot-backend/scripts/refresh_data.py --watch --ingest
```

## Important paths
- `frontend/` – chatbot widget & Nginx config.
- `chatbot-backend/app/` – FastAPI app, RAG graph, ingestion.
- `chatbot-backend/.env` – runtime config (not committed).
- `chatbot-backend/qdrant_data/` – persisted vector store.

## Troubleshooting
- Backend restarting: check `docker logs portfolio-backend` for missing env keys.
- Qdrant unhealthy: delete `chatbot-backend/qdrant_data/` and re‑ingest.
- Port conflicts: adjust published ports in `docker-compose.yml`.
