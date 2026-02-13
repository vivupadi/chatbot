from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
import time
from pathlib import Path
import uvicorn

from fastapi.responses import FileResponse

from src.ml.rag_updated import RAGengine
from src.data.ingestion import DocumentIngestion
from src.data.vectorstore import VectorStore
from src.utils.config import settings

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
from prometheus_client import Histogram  # ← Import for custom metrics

# Custom metric for initialization
initialization_duration = Gauge(
    'rag_initialization_duration_seconds',
    'Time taken to initialize RAG components'
)

#For custom Prometheus metrics
llm_latency = Histogram(
    'llm_inference_seconds',
    'Time for LLM inference',
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0]
)

#set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances (initialized on startup)
rag_engine: Optional[RAGengine] = None
vector_store: Optional[VectorStore] = None
document_ingestion: Optional[DocumentIngestion] = None


#Pydantic model      (Ensuring the that the input argumens are following the defined datatype)
class QueryRequest(BaseModel):
    question: str= Field(...,description="User Questions", min_length = 1 ) #fied()
    top_k : Optional[int] =Field(3, description='Number of documents to retrieve', ge=1, le=10)

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict]
    latency_seconds: float

class IngestURLRequest(BaseModel):
    url: str = Field(..., description="URL to scrape and ingest")

class IngestResponse(BaseModel):
    status: str
    documents_added: int
    message: str



#Startup event
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine, vector_store, document_ingestion   #Intialize everything on startup

    logger.info("=====Starting RAG Chatbot API=====")
    start_time = time.time()

    try:
        vector_store=VectorStore()
        rag_engine = RAGengine(vectorstore=vector_store)
        document_ingestion = DocumentIngestion()

        duration = time.time() - start_time
        initialization_duration.set(duration)
        logger.info(f"✅ RAG services initialized in {duration:.2f}s")
    except Exception as e:
        logger.error(f"Failed to initialize services:{e}")
        raise

    #App runs here
    yield


    ##Shutdown
    logger.info("Shutting down RAG Chatbot")
    
#Initilize FatAPI
app = FastAPI(
    title = 'RAG chatbot API',
    description= 'API for chatbot backend',
    version = '0.1',
    lifespan=lifespan
)

Instrumentator().instrument(app).expose(app)   #Prometheus metric initialized


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Query endpoint
@app.post("/query", response_model = QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        logger.info(f"Received query: {request.question}")
        
        # Measure LLM time
        start = time.time()
        #Query rag engine
        response = rag_engine.query(request.question)
        llm_latency.observe(time.time() - start)

        return QueryResponse(**response)
    except Exception as e:
        logger.error(f"Query failed:{e}")
        raise HTTPException(status_code=500, detail= f"Query failed{e}")
    

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return FileResponse("index.html")
    
    #return {
    #    "message": "RAG MLOps Chatbot API",
    #    "version": "0.1.0",
    #    "endpoints": {
    #        "query": "/query",
    #    }
    #}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
