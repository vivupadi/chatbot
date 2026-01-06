import pytest
import sys
from pathlib import Path
# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ml.llm_model import HuggingFaceInferenceLLM
from src.ml.rag_updated import RAGengine

def test_llm_model():
    LLM = HuggingFaceInferenceLLM()
    model = LLM.model
    assert model is not None


def test_rag_engine_initiated():
    rag = RAGengine()
    vector_store = rag.vector_store
    assert vector_store is not None