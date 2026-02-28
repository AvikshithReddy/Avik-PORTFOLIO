"""
FastAPI application orchestrating LangGraph + LlamaIndex RAG chatbot
for Avikshith Reddy's portfolio.
"""

import uuid
from typing import Dict, List
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
    SourceReference,
)
from app.llm.openai_client import OpenAIClient, CohereReranker
from app.rag import SYSTEM_PROMPT
from app.utils.logging import app_logger
from app.utils.text import truncate_text

# LangGraph / LlamaIndex stack
from app.rag.ingest import ingest_all
from app.rag.graph import build_graph
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Global instances
openai_client: OpenAIClient = None
reranker: CohereReranker = None
graph_app = None
retrievers: Dict[str, any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global openai_client, reranker, graph_app, retrievers

    app_logger.info("Starting Portfolio Chatbot API (LangGraph/LlamaIndex)...")
    settings.ensure_directories()

    openai_client = OpenAIClient(
        api_key=settings.OPENAI_API_KEY,
        chat_model=settings.CHAT_MODEL,
        embedding_model=settings.EMBEDDING_MODEL,
    )
    reranker = CohereReranker(
        api_key=settings.COHERE_API_KEY,
        model=settings.COHERE_RERANK_MODEL,
    )

    # Build retrievers backed by Qdrant collections
    retrievers = _build_retrievers()

    # Compile LangGraph
    graph_app = build_graph(
        router_fn=_route_message, retrievers=retrievers, reranker=reranker, synthesizer_fn=_synthesize
    )

    app_logger.info("Startup complete")
    yield
    app_logger.info("Shutting down Portfolio Chatbot API...")


def _build_retrievers():
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    embed_model = OpenAIEmbedding(model=settings.EMBEDDING_MODEL)
    retrievers = {}

    def make(collection):
        vs = QdrantVectorStore(client=client, collection_name=collection, enable_hybrid=settings.QDRANT_HYBRID)
        index = VectorStoreIndex.from_vector_store(vs, embed_model=embed_model)
        return index.as_retriever(similarity_top_k=settings.RAG_RETRIEVER_TOP_K)

    retrievers["resume"] = make(settings.QDRANT_RESUME_COLLECTION)
    retrievers["github"] = make(settings.QDRANT_GITHUB_COLLECTION)
    retrievers["portfolio"] = make(settings.QDRANT_PORTFOLIO_COLLECTION)
    retrievers["linkedin"] = make(settings.QDRANT_LINKEDIN_COLLECTION)
    # default fallback
    retrievers["general"] = retrievers["portfolio"]
    return retrievers


def _route_message(state):
    message = state.get("message", "").lower()
    if any(k in message for k in ["resume", "cv", "education", "experience", "work history"]):
        state["route"] = "resume"
    elif any(k in message for k in ["github", "repo", "code", "pull request", "commit"]):
        state["route"] = "github"
    elif any(k in message for k in ["linkedin", "profile", "recommendations", "endorsements"]):
        state["route"] = "linkedin"
    elif any(k in message for k in ["portfolio", "website", "site", "blog"]):
        state["route"] = "portfolio"
    else:
        state["route"] = "general"
    return state


def _format_sources(nodes: List) -> List[SourceReference]:
    sources = []
    for node in nodes:
        meta = node.metadata or {}
        source_type = meta.get("source_type", "unknown")
        locator = meta.get("file_path") or meta.get("url") or meta.get("source", "")
        name = (
            meta.get("source_name")
            or meta.get("title")
            or meta.get("repo")
            or meta.get("source_type", "source")
        )
        sources.append(
            SourceReference(
                source_type=source_type,
                source_name=name,
                locator=str(locator),
                snippet=truncate_text(node.get_text(), max_length=200),
                relevance_score=float(getattr(node, "score", 0.0) or 0.0),
            )
        )
    return sources


def _build_prompt(question: str, nodes: List) -> str:
    context_blocks = []
    per_node_limit = max(300, settings.RAG_MAX_CONTEXT_CHARS // max(1, len(nodes))) if nodes else settings.RAG_MAX_CONTEXT_CHARS
    for idx, node in enumerate(nodes, 1):
        meta = node.metadata or {}
        tag = meta.get("source_type", "source")
        node_text = truncate_text(node.get_text(), max_length=per_node_limit)
        context_blocks.append(f"[Source {idx}: {tag}]\n{node_text}")
    context = "\n\n".join(context_blocks)
    if not context:
        return f"No grounded context was found for this question. User question: {question}. Answer concisely or ask for a different question about Avikshith's work."
    return (
        f"CONTEXT:\n{context}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        "Answer concisely using only the context. Add inline tags like [Source: resume] or [Source: GitHub/<repo>] to cite evidence."
    )


def _synthesize(state):
    reranked = state.get("reranked", [])
    question = state.get("message", "")
    if not reranked:
        answer = "I don't have that in my indexed data yet. You can ask about my resume, GitHub projects, or portfolio site."
        state["answer"] = answer
        state["sources"] = []
        state["confidence"] = 0.0
        return state

    top_score = float(getattr(reranked[0], "score", 0.0) or 0.0)
    if top_score < settings.RAG_CONFIDENCE_THRESHOLD:
        answer = (
            "I couldn't find strong supporting information for that yet. "
            "Try asking about a specific project, role, or skill area, and I can look it up."
        )
        state["answer"] = answer
        state["sources"] = []
        state["confidence"] = top_score
        return state

    prompt = _build_prompt(question, reranked[: settings.RAG_TOP_K])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    answer = openai_client.chat_completion(
        messages=messages,
        temperature=settings.CHAT_TEMPERATURE,
        max_tokens=settings.CHAT_MAX_TOKENS,
    )

    sources = _format_sources(reranked[: settings.RAG_TOP_K])
    confidence = top_score

    state["answer"] = answer
    state["sources"] = sources
    state["confidence"] = confidence
    return state


# Create FastAPI app
app = FastAPI(
    title="Portfolio Chatbot API",
    description="LangGraph + LlamaIndex chatbot for Avikshith Reddy",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    loaded = graph_app is not None
    return HealthResponse(
        status="ok",
        version="2.0.0",
        rag_index_loaded=loaded,
        total_documents=0,
    )


async def verify_admin_key(x_admin_key: str = Header(None)):
    if settings.ADMIN_INGEST_KEY and x_admin_key != settings.ADMIN_INGEST_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return True


@app.post("/api/ingest", response_model=IngestResponse, dependencies=[Depends(verify_admin_key)])
async def ingest_documents(request: IngestRequest):
    try:
        result = ingest_all(request.sources, force_rebuild=request.force_rebuild)
        return IngestResponse(
            status=result.get("status", "unknown"),
            documents_processed=result.get("documents_processed", 0),
            chunks_created=result.get("chunks_created", 0),
            embeddings_generated=result.get("chunks_created", 0),
            sources_ingested=request.sources,
            errors=[],
        )
    except Exception as e:
        app_logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {e}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not graph_app:
        raise HTTPException(status_code=503, detail="RAG graph not initialized; run /api/ingest first")

    session_id = request.session_id or str(uuid.uuid4())

    state = graph_app.invoke(
        {"message": request.query},
        config={"configurable": {"thread_id": session_id}},
    )

    answer = state.get("answer", "I couldn't generate a response.")
    sources = state.get("sources", []) if request.include_sources else []
    confidence = float(state.get("confidence", 0.0))

    return ChatResponse(
        response=answer,
        session_id=session_id,
        confidence=confidence,
        sources=sources,
    )


@app.get("/")
async def root():
    return {
        "name": "Portfolio Chatbot API",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/api/chat",
            "ingest": "/api/ingest (admin only)",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL,
    )
