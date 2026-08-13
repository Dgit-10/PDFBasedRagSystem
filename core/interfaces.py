from abc import ABC, abstractmethod
from typing import List, Dict, Any

class ITextChunker(ABC):
    """Interface for text chunking strategies."""
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        pass

class IQueryTransformer(ABC):
    """Interface for query rewriting and transformation."""
    @abstractmethod
    def transform(self, query: str) -> str:
        pass

class IDocumentRanker(ABC):
    """Interface for re-ranking retrieved documents."""
    @abstractmethod
    def rank(self, query: str, documents: List[str], top_k: int) -> List[str]:
        pass

class IRetriever(ABC):
    """Interface for fetching documents from a database."""
    @abstractmethod
    def retrieve(self, query: str, limit: int) -> List[str]:
        pass