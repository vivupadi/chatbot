## Personalized RAG Based Chatbot
An Enterprise-grade chatbot to fetch answers about Vivek's Professional Life. Hosted on Hetzner Cloud using Kubernetes(K3s). (In progress)

The Chatbot uses hybrid approach to answer query:

- Main LLM model: Through Featherless API using Mistral 7b model

- Fallback model(in case featherless API fails): Ollama hosted 'tinyllama' model.

Configmap can be edited to select the architecture for the backend LLM model:
1. Hybrid Approach (Featherless API as main, Ollama 'tinyllama' model as fallback)
2. Featherless only (Only Featherless)
3. Ollama only (Only Ollama : for secure approach) 

[Live DEMO of the chatbot](http://188.245.71.76:30080/) 
(**Click on send button in the terminal)


## TECH STACK

- Python
- ChromaDB
- Langchain
- HuggingFace
- FeatherLess API
- FastAPI
- Docker
- Kubernetes
- Hetzner Cloud
- HTML/CSS
- Prometheus/Grafana
- PyTest

  
Multi Image approach: 

Frontend & Backend independent docker images

### Frontend

Files
- index.html
- nginx.conf

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

## Hosting & Scaling
Docker Image --> Hetzner Cloud --> Kubernetes(K3s)

K3 Manifests(yaml): 
- Namespace
- Secrets
- Backend:
  - Backend-depl
  - Backend-svc (type ClusterIP)
  - Configmap
- Frontend
  - Frontend-depl
  - Frontend-svc (type NodePort)
 
- Monitoring
  - Grafana-depl
  - Grafana-svc
  - Prometheus-depl
  - Prometheus-svc
  - Prometheus-configmap
 
## Monitoring 

Prometheues - For metrics and monitoring
Grafana - For Dashboard & Visualization

## Next Steps
### Evaluation

Current approach uses Golden Set Manual evaluations. Set of expected answers are compared with generated responses.
Next Plan:
RAGAS. Four core metrics:
Faithfulness — does the answer stick to the retrieved context, or is the LLM hallucinating? Score 0–1. Most important metric. If your answer says something not present in the retrieved chunks, faithfulness is low.
Answer Relevancy — does the answer actually address the question asked? A faithful but off-topic answer scores low here.
Context Precision — of the chunks you retrieved, how many were actually relevant? If you retrieve 5 chunks but only 1 was useful, precision is low. This tells you your retriever is noisy.
Context Recall — did you retrieve all the chunks needed to answer the question? If the answer requires information from 3 chunks but you only retrieved 1, recall is low.

### Cache using redis 


## LICENSES

This project is licensed under the MIT License - see the [License](LICENSE) file for details.



<div align="center">
⭐ Star this repo if you find it helpful!
