# RAG Based Chatbot
An Enterprise-grade chatbot to fetch answers about Vivek's Professional Life. Hosted and scaled on Oracle. 

## TECH STACK

### Frontend
### BackEnd
#### VectorStore
Chromadb is used as the vectorstore. The embedding models used was "sentence-transformers/all-MiniLM-L6-v2"
Chunking strategy initially used was fixed chunking with  chunk_size(300) and chunk_overlap(50) (Since the documents are sourced from personal cv and website portfolio).

What happens inside the vectorstore.py script???

Original Document

↓
    
[chunk_documents()]

↓
    
Text Chunks (still text!)

↓
    
[add_texts() calls embedding_function.embed_query()]

↓
    
Each chunk → HuggingFaceEmbeddings → Vector

↓
    
ChromaDB stores:
  - Text: "chunk content"
  - Vector: [0.1, 0.2, ...]
  - Metadata: {...}
    
#### LLM model
The LLM model used is mistralai/Mistral-7B-Instruct-v0.2 called using Featherless_ai(10$ paid plan).
### Hosting
### Scaling

## LICENSES
