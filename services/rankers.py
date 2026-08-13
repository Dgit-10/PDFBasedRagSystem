import json
import ollama
from typing import List
from core.interfaces import IDocumentRanker
from config import config
import re

class OllamaReRanker(IDocumentRanker):
    def rank(self, query: str, documents: List[str], top_k: int) -> List[str]:
        if not documents:
            return []
            
        prompt = f"Query: {query}\n\n"
        for i, doc in enumerate(documents):
            prompt += f"[Doc {i}]: {doc[:500]}...\n"
            
        prompt += "\nRate each document's relevance to the query from 0 to 10. Return ONLY a valid JSON list of integers representing the scores. Example: [8, 2, 10, 0]"
        
        try:
            response = ollama.chat(
                model=config.GENERATION_MODEL, 
                messages=[{'role': 'user', 'content': prompt}],
                format='json' # Forces Ollama to output valid JSON
            )
            
            content = response['message']['content'].strip()
            # Extract JSON array using regex in case of conversational padding
            match = re.search(r'\[.*\]', content, re.DOTALL)
            scores = json.loads(match.group()) if match else []
            
            # Pad with zeros if model outputs fewer scores than documents
            while len(scores) < len(documents):
                scores.append(0)
                
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, score in doc_scores[:top_k]]
        except Exception as e:
            print(f"Ranking failed, falling back to original order: {e}")
            return documents[:top_k]