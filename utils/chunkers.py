import re
from typing import List
from core.interfaces import ITextChunker

class RecursiveCharacterChunker(ITextChunker):
    def __init__(self, max_chunk_size: int = 1000, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        # Split priorities: Paragraphs -> Sentences -> Words
        self.separators = ["\n\n", "\n", "(?<=\. )", " "]

    def chunk(self, text: str) -> List[str]:
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        # Base case: if text is small enough, return it
        if len(text) <= self.max_chunk_size:
            return [text]
            
        separator = separators[0] if separators else ""
        for sep in separators:
            if re.search(sep, text):
                separator = sep
                break
                
        splits = re.split(separator, text)
        good_splits = []
        current_chunk = ""
        
        for split in splits:
            if len(current_chunk) + len(split) < self.max_chunk_size:
                current_chunk += split + separator
            else:
                if current_chunk:
                    good_splits.append(current_chunk.strip())
                current_chunk = split + separator
                
        if current_chunk:
            good_splits.append(current_chunk.strip())
            
        return good_splits