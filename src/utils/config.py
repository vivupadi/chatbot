from pydantic_settings import BaseSettings
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
    chunk_size = 300
    chunk_overlap = 50


    #LLM config
    use_ollama = True
    ollama_base_url = "http://localhost:11434"
    ollama_model = "llama3.2"

    #HugginFace
    huggingface_token = 'HuggingFace_Token'
    huggingface_model = "mistralai/Mistral-7B-Instruct-v0.2"

    # Vector Store
    vector_store_type: Literal["chroma", "faiss"] = "chroma"
    top_k_results = 3

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


#Global settings instance
settings = Settings()

# Create necessary directories
settings.huggingface_token = 'HuggingFace_Token'
settings.data_raw_dir.mkdir(parents=True, exist_ok=True)
settings.data_processed_dir.mkdir(parents=True, exist_ok=True)
settings.data_vector_dir.mkdir(parents=True, exist_ok=True)