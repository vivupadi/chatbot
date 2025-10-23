from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
from typing import Literal




class Settings(BaseSettings):
    #Project name
    project_name: str = "RAG Chatbot"
    environment: Literal["development", "staging", "production"] = "development"

    #Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    data_raw_dir: Path = Field(default_factory=lambda: Path("./data/raw"))
    data_processed_dir: Path = Field(default_factory = lambda: Path("./data/processed"))
    data_vector_dir: Path = Field(default_factory= lambda: Path("./data/vector_db"))


    #Embedding Model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"


    #Text processing
    chunk_size:int = 300
    chunk_overlap:int = 50


    #LLM config
    use_ollama:bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    #HugginFace
    huggingface_token: str = 'HuggingFace_Token'
    huggingface_model: str = "google/flan-t5-large"

    # Vector Store
    vector_store_type: Literal["chroma", "faiss"] = "chroma"
    top_k_results: int = 3

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields in .env
    )


#Global settings instance
# ========== GLOBAL INSTANCE ==========
def get_settings() -> Settings:
    """Factory function to create settings instance"""
    return Settings()

# Create global settings instance
settings = get_settings()

# Create necessary directories
settings.huggingface_token = 'HuggingFace_Token'
settings.data_raw_dir.mkdir(parents=True, exist_ok=True)
settings.data_processed_dir.mkdir(parents=True, exist_ok=True)
settings.data_vector_dir.mkdir(parents=True, exist_ok=True)