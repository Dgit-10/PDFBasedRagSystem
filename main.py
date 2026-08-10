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

@app.get("/health")
async def operational_health():
    return {"status": "ONLINE", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)