"""Application configuration using Pydantic Settings."""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """System settings and paths."""
    
    # App Info
    app_name: str = "Financial Audit Hybrid RAG"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # Base Directories
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data"
    sample_reports_dir: Path = data_dir / "sample_reports"
    evaluation_dir: Path = data_dir / "evaluation"
    chroma_persist_dir: Path = data_dir / "chroma_db"
    
    # API & Keys
    gemini_api_key: Optional[str] = None
    default_llm_model: str = "gemini-2.5-flash"
    default_embedding_model: str = "text-embedding-004"
    
    # Retrieval Hyperparameters
    dense_top_k: int = 10
    sparse_top_k: int = 10
    final_top_k: int = 5
    rrf_k: int = 60  # Standard Reciprocal Rank Fusion constant
    
    # Chunking Hyperparameters
    chunk_size: int = 800  # characters
    chunk_overlap: int = 150
    
    # Server Config
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

# Ensure directories exist
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.sample_reports_dir.mkdir(parents=True, exist_ok=True)
settings.evaluation_dir.mkdir(parents=True, exist_ok=True)
settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
