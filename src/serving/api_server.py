from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
import time
from pathlib import Path

from src.ml.rag_updated import RAGengine
from src.data.ingestion import DocumentIngestion
from src.data.vectorstore import VectorStore
from src.utils.config import settings

#set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#Initilize FatAPI
app = FastAPI(
    title = 'RAG chatbot API',
    description= 'API for chatbot backend',
    version = '0.1'
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
rag_engine: Optional[RAGengine] = None
vector_store: Optional[VectorStore] = None
document_ingestion: Optional[DocumentIngestion] = None