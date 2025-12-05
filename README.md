# RAG Based Chatbot
An Enterprise-grade chatbot to fetch answers about Vivek's Professional Life. Hosted and scaled on Oracle. (In progress)

Edit:
Hosting on 'Hetzner' for cost optimization.

Frontend & Backend docker images for independent approach.

## TECH STACK

### Frontend
Simple html-based UI that fetches the endpoints from FastAPI server.

![til](https://github.com/vivupadi/chatbot/blob/main/chatbot_snippet.jpg)

### BackEnd
#### VectorStore
Chromadb is used as the vectorstore. The embedding models used was "sentence-transformers/all-MiniLM-L6-v2"
Chunking strategy initially used was fixed chunking with  chunk_size(300) and chunk_overlap(50) (Since the documents are sourced from personal cv and website portfolio).

- What happens inside the vectorstore.py script???

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
The LLM model used is mistralai/Mistral-7B-Instruct-v0.2 called using Featherless_ai.

#### FastAPI
- Backend server to create API endpoints
- Receiving user questions
- Processing queries through the RAG system
- Returning AI-generated responses

Key Features:**
- Async/await support for high concurrency
- Built-in request/response validation
- CORS middleware for cross-origin requests
- Automatic interactive API docs at `/docs`
- Type-safe API development

### Hosting & Scaling
Docker Image --> Oracle Kubernetes free tier (in progress)

K8 Manifests(yaml): 
- Namespace
- ConfigMap
- Secrets
- Persistent Volume Claim
- Chroma DB
- Ingress
- Backend

## Monitoring And Evaluation

Prometheues and DeepEval(In progress) 

## LICENSES

This project is licensed under the MIT License - see the [License](LICENSE) file for details.



<div align="center">
⭐ Star this repo if you find it helpful!
