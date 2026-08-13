import os
import time
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from config import config
from services.rag_services import RAGService
from utils.metric_utils import calculate_token_costs

# --- Define API Contracts (Schemas) ---
class QueryIn(BaseModel):
    query: str
    limit: int = 3

class MetricsOut(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    execution_cost_usd: float

class QueryOut(BaseModel):
    answer: str
    metrics: MetricsOut

# --- Add this to your API Contracts (Schemas) ---
class SimilarityQueryIn(BaseModel):
    query_text: str
    limit: int = 5

# --- Init Framework ---
app = FastAPI(title=config.APP_NAME, version="2.0.0")

# Setup safe processing dirs on load
os.makedirs(config.PDF_STORAGE_DIR, exist_ok=True)

# Global Instance Provider via Dependency Injection
def get_rag_service() -> RAGService:
    return RAGService()

# --- HTTP Endpoints ---

@app.post("/api/v1/ingest", status_code=202)
async def trigger_pdf_ingestion(background_tasks: BackgroundTasks, service: RAGService = Depends(get_rag_service)):
    """
    Accepts ingestion request instantly and processes documents asynchronously 
    in a non-blocking background thread worker.
    """
    background_tasks.add_task(service.background_ingest_all_pdfs)
    return {"status": "processing", "message": "PDF pipeline running asynchronously in background."}

@app.post("/api/v1/chat", response_model=QueryOut)
async def semantic_chat_query(payload: QueryIn, service: RAGService = Depends(get_rag_service)):
    """Handles vector search queries and computes structural usage matrix tracing."""
    try:
        start_time = time.time()
        res = service.search_and_generate(payload.query, payload.limit)
        
        p_tokens = res["prompt_t"]
        c_tokens = res["comp_t"]
        total_tokens = p_tokens + c_tokens
        
        usd_cost = calculate_token_costs(p_tokens, c_tokens)
        
        return QueryOut(
            answer=res["answer"],
            metrics=MetricsOut(
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=total_tokens,
                execution_cost_usd=usd_cost
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Engine Error: {str(e)}")

@app.post("/api/v1/similar_sources")
async def get_similar_sources(payload: SimilarityQueryIn, service: RAGService = Depends(get_rag_service)):
    """Returns a list of PDF filenames that contain semantically similar content."""
    try:
        sources = service.find_similar_sources(payload.query_text, payload.limit)
        return {"similar_pdfs": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity Search Error: {str(e)}")


from fastapi import UploadFile, File
import shutil

# --- New Schema for Direct Chat ---
class DirectQueryIn(BaseModel):
    query: str
    context: str

# --- New HTTP Endpoints ---

@app.post("/api/v1/chat_direct", response_model=QueryOut)
async def direct_chat_query(payload: DirectQueryIn, service: RAGService = Depends(get_rag_service)):
    """Handles chat queries against a specific provided text context."""
    try:
        start_time = time.time()
        res = service.generate_from_context(payload.query, payload.context)
        
        p_tokens = res["prompt_t"]
        c_tokens = res["comp_t"]
        total_tokens = p_tokens + c_tokens
        
        usd_cost = calculate_token_costs(p_tokens, c_tokens)
        
        return QueryOut(
            answer=res["answer"],
            metrics=MetricsOut(
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=total_tokens,
                execution_cost_usd=usd_cost
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Engine Error: {str(e)}")

@app.post("/api/v1/ingest_single")
async def ingest_single_file(file: UploadFile = File(...), service: RAGService = Depends(get_rag_service)):
    """Saves a single uploaded PDF to the storage directory and embeds it."""
    file_path = os.path.join(config.PDF_STORAGE_DIR, file.filename)
    
    # Save the file to the dataset directory
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Trigger embedding process
    result = service.ingest_single_pdf(file_path)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
        
    return {"status": "success", "message": result["message"]}


@app.get("/health")
async def operational_health():
    return {"status": "ONLINE", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


