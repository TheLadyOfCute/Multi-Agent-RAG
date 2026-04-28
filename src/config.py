"""
Configuration management for Agentic RAG System.
"""

from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, ConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ===== API Keys =====
    dashscope_api_key: str = Field(..., description="DashScope API key for Qwen LLM and embeddings", min_length=10)
    dashscope_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="DashScope OpenAI-compatible base URL"
    )
    llm_provider: str = Field(default="dashscope", description="LLM provider (dashscope)")
    cohere_api_key: Optional[str] = Field(None, description="Cohere API key for reranking (optional)")
    
    # ===== Database Configuration =====
    database_url: str = Field(default="postgresql://localhost:5432/agentic_rag", description="PostgreSQL connection string")
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection string")
    chroma_persist_dir: str = Field(default="data/chroma_db", description="ChromaDB persistence directory")
    collection_name: str = Field(default="documents", description="ChromaDB collection name")
    bm25_index_path: str = Field(default="data/bm25_index.pkl", description="BM25 index file path")
    upload_dir: str = Field(default="data/uploads", description="Uploaded files directory")
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j Bolt URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="multirag_neo4j", description="Neo4j password")
    
    # ===== Model Configuration =====
    llm_model: str = Field(default="qwen3.6-plus", description="Qwen model to use")
    llm_temperature: float = Field(default=0.0, description="LLM temperature (0.0-1.0)", ge=0.0, le=1.0)
    llm_max_tokens: int = Field(default=4096, description="Maximum tokens for LLM response", gt=0)
    embedding_model: str = Field(default="text-embedding-v4", description="DashScope embedding model")
    embedding_dimension: int = Field(default=1536, description="Embedding vector dimension", gt=0)
    
    # ===== Chunking Configuration =====
    chunk_size: int = Field(default=500, description="Fallback chunk size in tokens (used when ChunkingAdvisorAgent fails)", gt=0)
    chunk_overlap: int = Field(default=50, description="Fallback overlap between chunks in tokens (used when ChunkingAdvisorAgent fails)", ge=0)
    
    # ===== Retrieval Configuration =====
    retrieval_top_k: int = Field(default=10, description="Number of chunks to retrieve", gt=0)    
    # ===== Agent Configuration =====
    validator_threshold: float = Field(default=0.7, description="Validation sufficiency threshold", ge=0.0, le=1.0)
    validator_max_retries: int = Field(default=2, description="Maximum retrieval retry attempts", ge=0)
    critic_max_iterations: int = Field(default=2, description="Maximum critic regeneration iterations", ge=1)
    
    # ===== Cache Configuration =====
    cache_enabled: bool = Field(default=True, description="Enable Redis caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds (1 hour default)", gt=0)
    
    # ===== System Configuration =====
    log_level: str = Field(default="INFO", description="Logging level")
    api_host: str = Field(default="127.0.0.1", description="Backend bind host for code-based startup")
    api_port: int = Field(default=8000, description="Backend bind port for code-based startup", gt=0, lt=65536)
    api_reload: bool = Field(default=True, description="Enable Uvicorn reload for code-based startup")
    max_file_size_mb: int = Field(default=50, description="Maximum upload file size in MB", gt=0)
    allowed_file_types: str = Field(default="pdf,docx,txt", description="Allowed upload file types (comma-separated)")
    
    # ===== Performance Configuration =====
    batch_size: int = Field(default=10, description="Batch size for embedding generation", gt=0)
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of: {valid_levels}")
        return v_upper
    
    def get_allowed_file_types_list(self) -> List[str]:
        """Get allowed file types as list."""
        return [ext.strip() for ext in self.allowed_file_types.split(",")]
    
    def get_database_config(self) -> dict:
        """Get database configuration as dictionary."""
        return {
            "url": self.database_url,
            "pool_size": 10,
            "max_overflow": 20,
            "pool_timeout": 30
        }
    
    def get_redis_config(self) -> dict:
        """Get Redis configuration as dictionary."""
        return {
            "url": self.redis_url,
            "decode_responses": True,
            "socket_timeout": 5,
            "socket_connect_timeout": 5
        }
    
    def get_chroma_config(self) -> dict:
        """Get ChromaDB configuration as dictionary."""
        return {
            "persist_directory": self.chroma_persist_dir,
            "embedding_dimension": self.embedding_dimension
        }
    
    def get_llm_config(self) -> dict:
        """Get LLM configuration as dictionary."""
        return {
            "model": self.llm_model,
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
            "api_key": self.dashscope_api_key,
            "base_url": self.dashscope_base_url
        }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get global settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global settings
    settings = Settings()
    return settings
