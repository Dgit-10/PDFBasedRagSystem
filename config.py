from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class ConfigReader(BaseSettings):
    APP_NAME: str = "PDF RAG Engine (Ollama Local)"
    OLLAMA_URL: str = "http://localhost:11434"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    GENERATION_MODEL: str = "llama3.1"
    
    # Changed path to avoid dimension crashes with the old Gemini database
    CHROMA_DB_PATH: str = "./chroma_db_ollama" 
    PDF_STORAGE_DIR: Path = Path("D:\PDFDataset\CompressedPDFDataset")
    
    model_config = SettingsConfigDict(env_file=".env")

config = ConfigReader()