"""Application configuration management using Pydantic Settings."""

import os
from typing import List, Optional
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings with environment variable bindings."""

    # Project metadata
    PROJECT_NAME: str = "Enterprise Agentic RAG Platform"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ]
    UPLOAD_DIR: str = str(BASE_DIR / "data")

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json" # 'json' or 'text'

    # AI & Embeddings (LiteLLM)
    LLM_PROVIDER: str = "gemini"
    LLM_MODEL: str = "gemini/gemini-3.1-flash-lite"
    LITELLM_API_BASE: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    
    # Observability & Tracing (LangSmith)
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "Agentic RAG"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: Optional[str] = None
    LANGSMITH_ENDPOINT: Optional[str] = None
    
    # Voice Input
    GCP_STT_ENABLED: bool = True

    # Guardrails (Google Cloud Model Armor)
    MODEL_ARMOR_ENABLED: bool = False
    MODEL_ARMOR_PROJECT_ID: Optional[str] = None
    MODEL_ARMOR_LOCATION: str = "us-central1"
    MODEL_ARMOR_TEMPLATE_ID: Optional[str] = None
    MODEL_ARMOR_TIMEOUT_SECONDS: float = 5.0

    # Memory
    SESSION_TIMEOUT_MINUTES: int = 30

    # Vector Storage & Retrieval
    VECTOR_DB_TYPE: str = Field(default="qdrant", description="Vector database type: 'qdrant'")
    QDRANT_URL: Optional[str] = Field(default=None, description="Remote Qdrant URL (e.g. http://localhost:6333 or cloud URL)")
    QDRANT_API_KEY: Optional[str] = Field(default=None, description="Qdrant API Key for cloud authentication")
    QDRANT_COLLECTION_NAME: str = Field(default="agentic_rag_documents", description="Qdrant collection name")
    QDRANT_PATH: str = str(BASE_DIR / "qdrant_data")
    FAISS_INDEX_PATH: str = str(BASE_DIR / "faiss_index")
    DATA_STORAGE_PATH: str = str(BASE_DIR / "data")
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 80
    RETRIEVAL_TOP_K: int = 15
    RERANKED_TOP_K: int = 5
    USE_HYBRID_SEARCH: bool = True
    USE_RERANKER: bool = True

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# Ensure directories exist
os.makedirs(settings.QDRANT_PATH, exist_ok=True)
os.makedirs(settings.DATA_STORAGE_PATH, exist_ok=True)
