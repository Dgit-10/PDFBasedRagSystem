from typing import List
from core.interfaces import IRetriever
import chromadb

class HybridChromaRetriever(IRetriever):
    def __init__(self, collection: chromadb.Collection):
        self.collection = collection

    def retrieve(self, query: str, limit: int) -> List[str]:
        # 1. Vector Search (Semantic)
        vector_results = self.collection.query(query_texts=[query], n_results=limit * 2)
        vector_docs = vector_results["documents"][0] if vector_results["documents"] else []
        
        # 2. Keyword Search (Simulated Sparse via metadata/contains filter if supported, 
        # or relying on Chroma's experimental keyword search. Here we use basic vector results 
        # and assume a secondary keyword system exists. For production, integrate BM25 here).
        # We will mock keyword docs by just leveraging the vector docs for this example's simplicity.
        keyword_docs = vector_docs[::-1] # Mocking a different order from keyword search
        
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        k = 60 # RRF constant
        
        for rank, doc in enumerate(vector_docs):
            rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (k + rank + 1)
            
        for rank, doc in enumerate(keyword_docs):
            rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (k + rank + 1)
            
        # Sort by RRF score descending
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_docs[:limit]]