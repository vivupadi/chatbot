import pytest
import json
#from src.utils.config import settings
from src.data.vectorstore import VectorStore

#Check data presence
def check_processed_and_vectorised_data_existence():
    # Load documents
    with open("./data/processed/documents.json", 'r', encoding='utf-8') as f:
        documents = json.load(f)
    assert documents is not None

def check_vectorstore_is_initialized():
    #from src.data.vectorstore import VectorStore
    vs = VectorStore()
    assert vs.embeddings is not None  #check for vector embedding model 
    assert vs.client is not None    #check is chroma is initialized
    assert vs.vector_store is not None    #