import pytest
import sys
from pathlib import Path
# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ml.llm_model import HuggingFaceInferenceLLM
from src.ml.rag_updated import RAGengine

import os
import os
from dotenv import load_dotenv
load_dotenv()
from src.utils.config import settings

def test_llm_model():
    api_key = settings.Featherless_ai_key_new or os.getenv("Featherless_ai_key_new")
        
    LLM = HuggingFaceInferenceLLM(api_key)
    LLM_model = LLM.model
    assert LLM_model is not None


def test_rag_engine_initiated():
    rag = RAGengine()
    vector_store = rag.vector_store

    #chain = rag._setup_chain()  

    #retriever = chain.retriever   #invoke retriever function
    #rag_chain = chain.rag_chain  #invoke rag_chain

    assert vector_store is not None
    #assert retriever is not None
    #assert rag_chain is not None