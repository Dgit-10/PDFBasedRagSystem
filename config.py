from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class ConfigReader(BaseSettings):
    APP_NAME: str = "PDF RAG Q&A Engine"
    GEMINI_API_KEY: str  # Pydantic will look for this in the .env file
    EMBEDDING_MODEL: str = "gemini-embedding-2"
    GENERATION_MODEL: str = "gemini-3-flash-preview"
    CHROMA_DB_PATH: str = "./chroma_db"
    PDF_STORAGE_DIR: Path = Path("D:\PDFDataset\CompressedPDFDataset")
    
    PRICE_PER_1M_INPUT_TOKENS: float = 0.30
    PRICE_PER_1M_OUTPUT_TOKENS: float = 2.50

    # Tells Pydantic to read from the .env file automatically
    model_config = SettingsConfigDict(env_file=".env")

config = ConfigReader()