import os
import glob
import time
from typing import Dict, Any, List
import chromadb
from chromadb.utils import embedding_functions
import ollama

from config import config
from core.interfaces import ITextChunker, IQueryTransformer, IDocumentRanker, IRetriever
from utils.chunkers import RecursiveCharacterChunker
from services.query_transformers import OllamaQueryTransformer
from services.rankers import OllamaReRanker
from services.retrievers import HybridChromaRetriever
from utils.pdf_utils import extract_text_from_pdf

class AdvancedRAGService:
    def __init__(self, 
                 chunker: ITextChunker = None, 
                 transformer: IQueryTransformer = None,
                 ranker: IDocumentRanker = None):
        
        self.db_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        
        # Native ChromaDB Ollama Integration
        self.embed_fn = embedding_functions.OllamaEmbeddingFunction(
            url=f"{config.OLLAMA_URL}/api/embeddings",
            model_name=config.EMBEDDING_MODEL,
        )
        
        self.collection = self.db_client.get_or_create_collection(
            name="pdf_knowledge_base_local",
            embedding_function=self.embed_fn
        )
        
        self.chunker = chunker or RecursiveCharacterChunker()
        self.transformer = transformer or OllamaQueryTransformer()
        self.ranker = ranker or OllamaReRanker()
        self.retriever = HybridChromaRetriever(self.collection)

    def ingest_text(self, text: str, source_name: str):
        chunks = self.chunker.chunk(text)
        for i, chunk in enumerate(chunks):
            self.collection.add(
                documents=[chunk],
                metadatas=[{"source": source_name, "chunk_index": i}],
                ids=[f"{source_name}_{i}"]
            )

    def background_ingest_all_pdfs(self):
        pdf_files = glob.glob(os.path.join(config.PDF_STORAGE_DIR, "*.pdf"))
        existing_metadatas = self.collection.get(include=["metadatas"])["metadatas"]
        indexed_files = {m["source"] for m in existing_metadatas if m}

        for file_path in pdf_files:
            filename = os.path.basename(file_path)
            if filename in indexed_files:
                continue
            try:
                print(f"Processing: {filename}")
                raw_text = extract_text_from_pdf(file_path)
                self.ingest_text(raw_text, filename)
            except Exception as e:
                print(f"Error indexing {filename}: {e}")

    def ingest_single_pdf(self, file_path: str) -> Dict[str, Any]:
        filename = os.path.basename(file_path)
        existing_metadatas = self.collection.get(include=["metadatas"])["metadatas"]
        indexed_files = {m["source"] for m in existing_metadatas if m}
        
        if filename in indexed_files:
            return {"status": "skipped", "message": "File already exists in dataset."}
            
        try:
            raw_text = extract_text_from_pdf(file_path)
            self.ingest_text(raw_text, filename)
            return {"status": "success", "message": f"{filename} successfully embedded."}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def search_and_generate(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        optimized_query = self.transformer.transform(query)
        retrieved_docs = self.retriever.retrieve(optimized_query, limit=10)
        
        if not retrieved_docs:
            return {"answer": "No relevant domain data found.", "prompt_t": 0, "comp_t": 0}

        best_docs = self.ranker.rank(query, retrieved_docs, top_k=n_results)
        context_block = "\n\n".join(best_docs)
        prompt = f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer cleanly using only facts above:"
        
        response = ollama.chat(
            model=config.GENERATION_MODEL, 
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        return {
            "answer": response['message']['content'],
            "prompt_t": response.get('prompt_eval_count', 0),
            "comp_t": response.get('eval_count', 0)
        }

    def generate_from_context(self, query: str, context: str) -> Dict[str, Any]:
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer cleanly using only facts above:"
        response = ollama.chat(
            model=config.GENERATION_MODEL, 
            messages=[{'role': 'user', 'content': prompt}]
        )
        return {
            "answer": response['message']['content'],
            "prompt_t": response.get('prompt_eval_count', 0),
            "comp_t": response.get('eval_count', 0)
        }

    def find_similar_sources(self, query_text: str, limit: int = 5) -> List[str]:
        db_results = self.collection.query(query_texts=[query_text], n_results=limit * 3)
        unique_sources = []
        if db_results["metadatas"] and db_results["metadatas"][0]:
            for meta in db_results["metadatas"][0]:
                source_name = meta["source"]
                if source_name not in unique_sources:
                    unique_sources.append(source_name)
                if len(unique_sources) >= limit:
                    break
        return unique_sources