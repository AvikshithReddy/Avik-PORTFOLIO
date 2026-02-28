"""
Configuration management for the chatbot backend
"""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # OpenAI / model configuration
    OPENAI_API_KEY: str = ""
    CHAT_MODEL: str = "gpt-4o"
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    CHAT_TEMPERATURE: float = 0.2
    CHAT_MAX_TOKENS: int = 800

    # RAG Configuration
    RAG_INDEX_DIR: str = "./rag_index"  # legacy local index dir (not used with Qdrant)
    RAG_RETRIEVER_TOP_K: int = 12
    RAG_RERANK_TOP_N: int = 8
    RAG_TOP_K: int = 6
    RAG_CHUNK_SIZE: int = 1100
    RAG_CHUNK_OVERLAP: int = 150
    RAG_CODE_CHUNK_LINES: int = 120
    RAG_CODE_CHUNK_OVERLAP: int = 20
    RAG_CONFIDENCE_THRESHOLD: float = 0.55
    RAG_MAX_CONTEXT_CHARS: int = 12000

    # Source Paths
    PORTFOLIO_JSON_PATH: str = "./data/portfolio_data.json"
    RESUME_PDF_PATH: str = "./data/avikshithReddy_resume.pdf"
    PORTFOLIO_MD_GLOB: str = "./data/*.md"
    LINKEDIN_PDF_PATH: str = "./data/Profile.pdf"

    # GitHub Integration
    GITHUB_TOKEN: str = ""
    GITHUB_USERNAME: str = "avikshithreddy"
    GITHUB_REPOS: str = ""  # Comma-separated repo names; empty = top repos
    GITHUB_MAX_REPOS: int = 15
    GITHUB_BRANCH: str = "main"

    # Portfolio site ingest
    PORTFOLIO_URL: str = "https://avikshithreddy.github.io"
    LINKEDIN_DATA_DIR: str = "./data/linkedin"

    # Vector DB (Qdrant)
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_RESUME_COLLECTION: str = "portfolio_resume"
    QDRANT_GITHUB_COLLECTION: str = "portfolio_github"
    QDRANT_PORTFOLIO_COLLECTION: str = "portfolio_site"
    QDRANT_LINKEDIN_COLLECTION: str = "portfolio_linkedin"
    QDRANT_HYBRID: bool = True

    # Reranker
    COHERE_API_KEY: str = ""
    COHERE_RERANK_MODEL: str = "rerank-3.5"

    # Security
    ADMIN_INGEST_KEY: str = ""
    CORS_ALLOW_ORIGINS: str = "*"

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "info"

    # Chat session memory (server-side)
    CHAT_SESSION_MAX_SESSIONS: int = 200
    CHAT_SESSION_TTL_SECONDS: int = 6 * 60 * 60
    CHAT_SESSION_MAX_MESSAGES: int = 20
    CHAT_HISTORY_MESSAGES: int = 6

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

    @property
    def cors_origins(self) -> List[str]:
        if self.CORS_ALLOW_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ALLOW_ORIGINS.split(",")]

    @property
    def rag_index_path(self) -> Path:
        return Path(self.RAG_INDEX_DIR)

    def ensure_directories(self):
        for path in [
            self.rag_index_path,
            Path("./data"),
            Path("./data/resume"),
            Path(self.LINKEDIN_DATA_DIR),
            Path("./logs"),
        ]:
            if path.exists():
                continue
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError:
                # In containers the /app/data mount may be read-only; skip creating directories there.
                pass


settings = Settings()
