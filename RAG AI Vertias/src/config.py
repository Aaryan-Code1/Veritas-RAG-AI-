import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # App Environment
    ENVIRONMENT: str = Field(default="development", description="Runtime environment")
    LOG_LEVEL: str = Field(default="INFO", description="Logging verbosity")
    APP_HOST: str = Field(default="0.0.0.0", description="FastAPI host")
    APP_PORT: int = Field(default=8000, description="FastAPI port")

    # LLM Settings
    DEFAULT_LLM_PROVIDER: str = Field(default="anthropic", description="anthropic or openai")
    ANTHROPIC_API_KEY: str = Field(default="", description="Anthropic API Key")
    ANTHROPIC_MODEL: str = Field(default="claude-3-5-sonnet-20241022", description="Claude model identifier")
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API Key")
    OPENAI_MODEL: str = Field(default="gpt-4o", description="OpenAI model identifier")

    # Embedding Settings
    EMBEDDING_PROVIDER: str = Field(default="openai", description="Embedding provider")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-large", description="Embedding model name")
    EMBEDDING_DIMENSIONS: int = Field(default=1536, description="Embedding vector dimensionality")

    # Reranker Settings
    RERANKER_PROVIDER: str = Field(default="cohere", description="Reranker provider")
    COHERE_API_KEY: str = Field(default="", description="Cohere API Key")
    COHERE_RERANK_MODEL: str = Field(default="rerank-v3.5", description="Cohere model")
    RERANKER_CONFIDENCE_THRESHOLD: float = Field(
        default=0.45, description="Minimum rerank score to proceed with generation"
    )

    # Qdrant Database Settings
    QDRANT_HOST: str = Field(default="localhost", description="Qdrant hostname")
    QDRANT_PORT: int = Field(default=6333, description="Qdrant port")
    QDRANT_API_KEY: str = Field(default="", description="Qdrant API Key (if authenticated)")
    QDRANT_COLLECTION_NAME: str = Field(default="veritas_enterprise_v1", description="Collection name")

    # Retrieval & Generation Hyperparameters
    DEFAULT_CHUNK_SIZE: int = Field(default=512, description="Target tokens per chunk")
    DEFAULT_CHUNK_OVERLAP: int = Field(default=64, description="Token overlap between chunks")
    MAX_CONTEXT_TOKENS: int = Field(default=3500, description="Maximum context tokens passed to LLM")
    TOP_K_RETRIEVAL: int = Field(default=50, description="Initial candidates from hybrid search")
    FINAL_CONTEXT_CHUNKS: int = Field(default=7, description="Final top-N chunks passed to prompt")
    MMR_LAMBDA: float = Field(default=0.75, description="MMR diversity trade-off (0.0 to 1.0)")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
