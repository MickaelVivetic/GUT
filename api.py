from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
import tempfile
import os

from main import RAGAgent
from ingestion import TextIngestion


agent: Optional[RAGAgent] = None
ingestion_instances: dict = {}


def get_ingestion(client_id: str) -> TextIngestion:
    if client_id not in ingestion_instances:
        ingestion_instances[client_id] = TextIngestion(client_id=client_id)
    return ingestion_instances[client_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    agent = RAGAgent()
    yield
    agent = None


app = FastAPI(
    title="GUT RAG API",
    description="API pour le systeme RAG avec LangChain et Ollama",
    version="1.0.0",
    lifespan=lifespan
)


class QueryRequest(BaseModel):
    question: str
    client_id: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    question: str
    client_id: str


class IngestTextRequest(BaseModel):
    text: str
    client_id: str
    source: str = "api"


class IngestResponse(BaseModel):
    message: str
    client_id: str
    chunks_count: int


class HealthResponse(BaseModel):
    status: str
    message: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", message="Service is running")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        answer = agent.query(request.question, request.client_id, request.top_k)
        return QueryResponse(answer=answer, question=request.question, client_id=request.client_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(request: IngestTextRequest):
    try:
        ingestion = get_ingestion(request.client_id)
        chunks_count = ingestion.ingest_text(request.text, request.source)
        return IngestResponse(
            message="Text ingested successfully",
            client_id=request.client_id,
            chunks_count=chunks_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(client_id: str, file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        ingestion = get_ingestion(client_id)
        chunks_count = ingestion.ingest_file(tmp_path)
        
        os.unlink(tmp_path)
        
        return IngestResponse(
            message=f"File '{file.filename}' ingested successfully",
            client_id=client_id,
            chunks_count=chunks_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "GUT RAG API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
