# Portfolio Chatbot Backend

FastAPI backend for the portfolio chatbot. It answers questions using local portfolio content, resume data, markdown notes, and GitHub repositories via lightweight RAG.

Use Python 3.11 for local development. The current dependency lock was verified on Python 3.11 and the provided Docker image also uses Python 3.11.

## What Changed

- Startup now auto-builds the RAG index when one is not present.
- GitHub ingestion works with public repositories even without a token.
- Follow-up questions are rewritten into standalone retrieval queries before search.
- The assistant no longer treats prior assistant replies as grounding context.
- The frontend now fails clearly when the backend URL is missing instead of making broken same-origin calls.

## Architecture

```text
Static portfolio page
  -> frontend/chatbot.js
  -> POST /api/chat
FastAPI backend
  -> rewrite follow-up query if needed
  -> embed rewritten query
  -> retrieve relevant chunks from local RAG index
  -> answer only from retrieved context
Sources
  -> data/portfolio_data.json
  -> data/avikshithReddy_resume.pdf
  -> data/*.md
  -> GitHub repos / READMEs
```

## Local Setup

1. Create the backend environment file.

```bash
cd chatbot-backend
cp .env.example .env
```

2. Edit `.env`.

```env
OPENAI_API_KEY=sk-...
CHAT_MODEL=gpt-4.1-mini
QUERY_REWRITE_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small
AUTO_INGEST_ON_STARTUP=true
STARTUP_INGEST_SOURCES=portfolio,resume,markdown,github
GITHUB_USERNAME=avikshithreddy
GITHUB_TOKEN=
ADMIN_INGEST_KEY=change-me
CORS_ALLOW_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
```

`GITHUB_TOKEN` is optional. Without it, public repositories still ingest, but GitHub rate limits are lower.

3. Install and run the backend.

```bash
cd chatbot-backend
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

If `AUTO_INGEST_ON_STARTUP=true`, the index is built automatically on first boot. You do not need to call `/api/ingest` unless you want to rebuild manually.

4. Test the backend.

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What are Avikshith'\''s main projects?","include_sources":true}'
```

## Frontend Wiring

The widget is already loaded from `frontend/chatbot.js`. It now resolves the backend URL like this:

- `window.chatbotConfig.backendUrl` if you set one
- `http://localhost:8000` when opened locally
- no backend by default in production, with an explicit status message

For production, set the deployed backend URL in `index.html`:

```html
<script>
  window.CHATBOT_BACKEND_URL = 'https://YOUR-CLOUD-RUN-URL';
</script>
```

That must appear before the existing `window.chatbotConfig` block.

## Manual Rebuild

Use this only when you want to force a fresh index.

```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: change-me" \
  -d '{"sources":["portfolio","resume","markdown","github"],"force_rebuild":true}'
```

## Docker

Local Docker is optional. Use it when you want a portable backend container.

```bash
cd chatbot-backend
docker compose up --build
```

The container respects `$PORT`, auto-builds the RAG index on startup, and mounts `rag_index/` and `logs/` locally through Compose.

## Google Cloud Run

Use Cloud Run only for the backend. Keep the portfolio itself on GitHub Pages or your current static host.

### Option 1: Deploy From Cloud Build

The repo includes `cloudbuild.yaml`, which now builds from the correct Dockerfile and enables startup ingestion.

From `chatbot-backend/`:

```bash
gcloud builds submit --config cloudbuild.yaml .
```

Before running that, create these Secret Manager secrets in your Google Cloud project:

- `OPENAI_API_KEY`
- `GITHUB_TOKEN`
- `ADMIN_INGEST_KEY`

`GITHUB_TOKEN` can be empty if you only need public repositories.

### Option 2: Manual Cloud Run Deploy

```bash
cd chatbot-backend
gcloud builds submit --tag gcr.io/YOUR_PROJECT/avik-portfolio-chatbot .

gcloud run deploy avik-portfolio-chatbot \
  --image gcr.io/YOUR_PROJECT/avik-portfolio-chatbot \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000 \
  --set-env-vars OPENAI_API_KEY=sk-...,GITHUB_USERNAME=avikshithreddy,AUTO_INGEST_ON_STARTUP=true,STARTUP_INGEST_SOURCES=portfolio,resume,markdown,github,CORS_ALLOW_ORIGINS=https://YOUR-GITHUB-PAGES-DOMAIN
```

If you want GitHub ingestion beyond anonymous rate limits, also add `GITHUB_TOKEN=...`.

After deployment, copy the Cloud Run service URL and set it in `window.CHATBOT_BACKEND_URL` inside `index.html`.

## API

### `GET /health`

```json
{
  "status": "ok",
  "version": "1.0.0",
  "rag_index_loaded": true,
  "total_documents": 42
}
```

### `POST /api/chat`

Request:

```json
{
  "query": "Tell me about Avikshith's finance chatbot project",
  "session_id": "optional-session-id",
  "include_sources": true
}
```

Response:

```json
{
  "response": "Avikshith built a finance-focused LLM assistant...",
  "session_id": "abc-123",
  "confidence": 0.81,
  "answer_mode": "grounded",
  "sources": [
    {
      "source_type": "portfolio",
      "source_name": "Project: Financial Chatbot",
      "locator": "./data/portfolio_data.json",
      "snippet": "Built a finance-focused LLM assistant..."
    }
  ]
}
```

`answer_mode` is the user-facing signal to trust, not raw `confidence`. The retrieval score is still returned for debugging and tuning.

## Verification

Run the unit/smoke tests:

```bash
cd chatbot-backend
python -m unittest discover -s tests
```

## Recommended Deployment Workflow

1. Keep the static portfolio in GitHub Pages or your current Git-based static host.
2. Deploy only `chatbot-backend/` to Cloud Run.
3. Set the Cloud Run URL in `window.CHATBOT_BACKEND_URL` in `index.html`.
4. Commit and push the static portfolio update.
5. Verify `/health` and one real `/api/chat` request after deploy.
