import ollama
from core.interfaces import IQueryTransformer
from config import config

class OllamaQueryTransformer(IQueryTransformer):
    def transform(self, query: str) -> str:
        prompt = (
            "You are a search query rewriting expert. "
            "Rewrite the following user query to be highly descriptive, expanding on keywords "
            "and technical terms to improve vector database retrieval. "
            "Output ONLY the new query, nothing else.\n\n"
            f"Original Query: {query}"
        )
        response = ollama.chat(
            model=config.GENERATION_MODEL, 
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content'].strip()