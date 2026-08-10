import os
import glob
from typing import List, Dict, Any
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from google import genai

from config import config
from utils.pdf_utils import extract_text_from_pdf, chunk_text
from utils.metric_utils import calculate_token_costs
import time

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def __call__(self, input: Documents) -> Embeddings:
        response = self.client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=input
        )
        return [emb.values for emb in response.embeddings]


class RAGService:
    def __init__(self):
        print("Initializing the Gemini Model RAG client")
        print(config.GENERATION_MODEL)
        print(config.model_dump())
        # Initialize Google GenAI Client
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        
        # Initialize ChromaDB Local Engine
        self.db_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        self.embed_fn = GeminiEmbeddingFunction(api_key=config.GEMINI_API_KEY)
        self.collection = self.db_client.get_or_create_collection(
            name="pdf_knowledge_base",
            embedding_function=self.embed_fn
        )


    # Inside RAGService class:
    def background_ingest_all_pdfs(self):
        pdf_files = glob.glob(os.path.join(config.PDF_STORAGE_DIR, "*.pdf"))
        
        # 1. Get list of already indexed files to avoid duplicates
        existing_metadatas = self.collection.get(include=["metadatas"])["metadatas"]
        indexed_files = {m["source"] for m in existing_metadatas}

        for file_path in pdf_files:
            filename = os.path.basename(file_path)
            if filename in indexed_files:
                continue  # Skip already processed files
            
            try:
                print(f"Processing: {filename}")
                raw_text = extract_text_from_pdf(file_path)
                chunks = chunk_text(raw_text)
                
                # 2. Add in small batches to respect RPM (Requests Per Minute)
                # Embedding 1,078 files can be slow; keep chunks per file small
                for i, chunk in enumerate(chunks):
                    self.collection.add(
                        documents=[chunk],
                        metadatas=[{"source": filename, "chunk_index": i}],
                        ids=[f"{filename}_{i}"]
                    )
                
                # 3. Rate Limit Management: Pause slightly between files
                time.sleep(6) # 6 seconds sleep = 10 files per minute, safe for free tier
                
            except Exception as e:
                print(f"Error indexing {filename}: {e}")

    def search_and_generate(self, query: str, n_results: int) -> Dict[str, Any]:
        """Queries database context and formats request tracing."""
        db_results = self.collection.query(query_texts=[query], n_results=n_results)
        print("DB Result Generated ",db_results)
        retrieved_docs = db_results["documents"][0] if db_results["documents"] else []
        sources = db_results["metadatas"][0] if db_results["metadatas"] else []
        
        if not retrieved_docs:
            return {"answer": "No relevant domain data found.", "prompt_t": 0, "comp_t": 0}

        context_block = "\n\n".join(
            [f"[File: {meta['source']}]\n{doc}" for doc, meta in zip(retrieved_docs, sources)]
        )

        prompt = f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer cleanly using only facts above:"
        
        response = self.client.models.generate_content(
            model=config.GENERATION_MODEL,
            contents=prompt
        )
        
        meta = response.usage_metadata
        return {
            "answer": response.text,
            "prompt_t": meta.prompt_token_count if meta else 0,
            "comp_t": meta.candidates_token_count if meta else 0
        }