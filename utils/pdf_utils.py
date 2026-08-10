import os
from typing import List
from pypdf import PdfReader

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts raw string text from a local PDF file path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target PDF file not found at {file_path}")
        
    reader = PdfReader(file_path)
    full_text = []
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            full_text.append(extracted)
    return "\n".join(full_text)

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Splits text content into granular overlapping segments."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks