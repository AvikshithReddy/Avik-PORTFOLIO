"""
Configuration management for the chatbot backend
"""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str
    CHAT_MODEL: str = "gpt-4-turbo-preview"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHAT_TEMPERATURE: float = 0.2
    CHAT_MAX_TOKENS: int = 1000
    
    # RAG Configuration
    RAG_INDEX_DIR: str = "./rag_index"
    RAG_TOP_K: int = 6
    RAG_CHUNK_SIZE: int = 1200
    RAG_CHUNK_OVERLAP: int = 150
    RAG_CONFIDENCE_THRESHOLD: float = 0.55
    
    # Source Paths
    PORTFOLIO_JSON_PATH: str = "./data/portfolio_data.json"
    RESUME_PDF_PATH: str = "./data/avikshithReddy_resume.pdf"
    PORTFOLIO_MD_GLOB: str = "./data/*.md"
    
    # GitHub Integration
    GITHUB_TOKEN: str = ""
    GITHUB_USERNAME: str = "avikshithreddy"
    GITHUB_MAX_REPOS: int = 20
    
    # Security
    ADMIN_INGEST_KEY: str = ""
    CORS_ALLOW_ORIGINS: str = "*"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "info"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env
    
    @property
    def cors_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string"""
        if self.CORS_ALLOW_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ALLOW_ORIGINS.split(",")]
    
    @property
    def rag_index_path(self) -> Path:
        """Get RAG index directory as Path object"""
        return Path(self.RAG_INDEX_DIR)
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        self.rag_index_path.mkdir(parents=True, exist_ok=True)
        Path("./data").mkdir(parents=True, exist_ok=True)
        Path("./logs").mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
