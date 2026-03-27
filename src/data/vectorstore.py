import chromadb
from chromadb.config import Settings as ChromaSettings

from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional
import logging
from pathlib import Path
from src.utils.config import settings

from rank_bm25 import BM25Okapi

import json
import os
from dotenv import load_dotenv
load_dotenv(override=False)

from azure.storage.blob import BlobServiceClient

from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        logger.info("Initialize Embeddings..")
        """self.embeddings = HuggingFaceEmbeddings(
            model_name = settings.embedding_model,
            model_kwargs = {'device': settings.embedding_device},
            encode_kwargs={'normalize_embeddings': True}
        )"""

        # Use HuggingFace API for embeddings (saves 300MB RAM!)
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",  # Same model!
            huggingfacehub_api_token=settings.HuggingFace_Token
        )

        logger.info("🔧 Initializing ChromaDB...")

        running_mode = settings.running_mode   #Or local   Set running mode here to create a vector database on hetzner cloud or locally
        self.chroma_path= "/app/data/vector_db"

        if running_mode == 'local':
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
            path=str(settings.data_vector_dir),
            settings=ChromaSettings(anonymized_telemetry=False)  # Disables telemetry
            )
        elif running_mode == 'cloud':
            self.client = chromadb.PersistentClient(
            path=self.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            client=self.client,
            collection_name="rag_documents",
            embedding_function=self.embeddings
        )

        logger.info("Vector store initialized")

        CONNECT_STR = os.getenv("CONNECTION_STRING")
        CONTAINER_NAME = os.getenv("CONTAINER_NAME")

        self.blob_service = BlobServiceClient.from_connection_string(CONNECT_STR)
        self.container = self.blob_service.get_container_client(CONTAINER_NAME)

        logger.info("Azure Blob initialized")

    def chunk_documents(self,documents: List[Dict]):
        chunk_size = settings.chunk_size
        chunk_overlap = settings.chunk_overlap

        logger.info("Chunking Started..(size={chunk_size}, overlap={chunk_overlap})")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap  = chunk_overlap,
            length_function = len,
            separators = [
                "\n\n", 
                "\nPersonal Information", 
                "\nWork Experience", 
                "\nResearch and Projects",
                "\nEducation",
                "\nCore Skills", 
                "\nLanguages",
                "\nCourses and Licenses",
                "\no ", 
                " ", 
                ""
                ]
        )

        chunked_docs = []
        for doc in documents:
            chunks = text_splitter.split_text(doc['content'])

            for i, chunk in enumerate(chunks):
                chunked_doc = {
                    'content': chunk,
                    'metadata': {
                        **doc['metadata'],
                        'chunk_id': i,
                        'total_chunks': len(chunks)

                    }
                }
                chunked_docs.append(chunked_doc)
        
        logger.info(f"✅ Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs
    
    def add_documents(self, documents, do_chunk = True):   #Add documents to vector store, do_chunk: Whether to chunk documents first
        if do_chunk:
            documents = self.chunk_documents(documents)
        
        logger.info(f"Adding {len(documents)} documents to vector store...")
        
        # Prepare for ChromaDB
        texts = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        # Add to vector store     
        self.vector_store.add_texts(       # Vectorization
            texts=texts,
            metadatas=metadatas
        )
        
        logger.info("✅ Documents added to vector store")

    def load_json(self):  #Extract text from pdf
        try:
            blob_path = "json/documents.json"
            logger.info(f"PDF Loading: {blob_path}")
            json_bytes = self.container.get_blob_client(blob_path).download_blob().readall()   #gets pdf file bytes format
            
            json_string = json_bytes.decode('utf-8')
    
            document = json.loads(json_string)
            return document

        except Exception as e:
            logger.error(f" Error Loading pdf: {e}")
            return []

    def upload_chroma_to_blob(self):
        """Upload chromadb to blob"""
        print("Uploading chromadb to blob...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        uploaded_count = 0
        
        # Upload all ChromaDB files from the directory
        for file_path in Path(self.chroma_path).rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(self.chroma_path)
                
                # Create blob names
                versioned_blob = f"vectorstore/versions/{timestamp}/{relative_path}"
                latest_blob = f"vectorstore/latest/{relative_path}"
                
                # Read file and upload
                with open(file_path, 'rb') as f:
                    file_bytes = f.read()

                # Upload versioned backup
                self.container.get_blob_client(versioned_blob).upload_blob(
                    file_bytes, 
                    overwrite=True
                )
                
                # Upload to latest
                self.container.get_blob_client(latest_blob).upload_blob(
                    file_bytes,
                    overwrite=True
                )
                
                uploaded_count += 1
        
        print("✓ Uploaded chroma to blob")


    def similarity_search(self, query, k=None, filter_metadata=None):      #Search for similar documents
        k = settings.top_k_results

        logger.info(f"🔍 Searching for: '{query}' (top_k={k})")

        results = self.vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            filter=filter_metadata
        )

        #Foramt results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'relevance_score': score
            })
        
        logger.info(f"FOund {len(formatted_results)} relevant documents")
        return formatted_results

    def get_retriever(self, search_kwargs: Optional[Dict] = None):   #Get LangChain retriever
        
        search_kwargs = search_kwargs or {"k": settings.top_k_results}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def delete_collection(self):
        """Delete the entire collection"""
        logger.warning("⚠️  Deleting vector store collection...")
        self.client.delete_collection("rag_documents")
        logger.info("✅ Collection deleted")

    def _init_bm25(self):
        """Initialize BM25 index from existing documents"""
        logger.info("🔧 Initializing BM25 index...")
        
        try:
            # Get all documents from ChromaDB
            all_docs = self.vector_store.get()
            
            if not all_docs or not all_docs.get('documents'):
                logger.warning("⚠️ No documents found for BM25 indexing")
                self.bm25 = None
                self.bm25_docs = []
                self.bm25_metadatas = []
                return
            
            # Store documents and metadata
            self.bm25_docs = all_docs['documents']
            self.bm25_metadatas = all_docs.get('metadatas', [{}] * len(self.bm25_docs))
            
            # Tokenize documents for BM25 (simple whitespace tokenization)
            tokenized_docs = [doc.lower().split() for doc in self.bm25_docs]
            
            # Create BM25 index
            self.bm25 = BM25Okapi(tokenized_docs)
        
            logger.info(f"✅ BM25 index created with {len(self.bm25_docs)} documents")
            
        except Exception as e:
            logger.error(f"❌ BM25 initialization failed: {e}")
            self.bm25 = None
            self.bm25_docs = []
            self.bm25_metadatas = []

    def hybrid_search(self, query: str, k: int = 10, alpha: float = 0.5):
        """
        Hybrid search combining BM25 (keyword) + Vector (semantic)
        """
        # Initialize BM25 if not already done
        if not hasattr(self, 'bm25') or self.bm25 is None:
            self._init_bm25()
        
        # If BM25 not available, fallback to vector-only
        if self.bm25 is None:
            logger.warning("⚠️ BM25 not available, using vector search only")
            return self.vector_store.similarity_search(query, k=k)
        
        # 1. BM25 search (keyword-based)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores to [0, 1]
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_scores_norm = [score / max_bm25 for score in bm25_scores]
        
        # 2. Vector search (semantic)
        vector_results = self.vector_store.similarity_search_with_score(query, k=len(self.bm25_docs))
        
        # Create mapping: content -> (vector_score, doc)
        vector_score_map = {}
        for doc, score in vector_results:
            # ChromaDB uses distance (lower is better), convert to similarity
            similarity = 1 / (1 + score)  # Convert distance to similarity [0, 1]
            vector_score_map[doc.page_content] = (similarity, doc)
        
        # 3. Combine scores using alpha weighting
        combined_scores = []
        for i, (doc_content, metadata) in enumerate(zip(self.bm25_docs, self.bm25_metadatas)):
            bm25_score = bm25_scores_norm[i]
            
            # Get vector score if available
            if doc_content in vector_score_map:
                vector_score, doc_obj = vector_score_map[doc_content]
            else:
                vector_score = 0
                # Create document object
                from langchain_core.documents import Document
                doc_obj = Document(page_content=doc_content, metadata=metadata)
            
            # Hybrid score: weighted combination
            hybrid_score = (1 - alpha) * bm25_score + alpha * vector_score
            
            combined_scores.append((hybrid_score, doc_obj))
        
        # 4. Sort by hybrid score and return top k
        combined_scores.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in combined_scores[:k]]
        
        logger.info(f"✅ Hybrid search: returned {len(top_docs)} documents (α={alpha})")
        
        return top_docs


# Example usage
if __name__ == "__main__":
    import json
    
    # Initialize vector store
    vs_manager = VectorStore()

    documents = vs_manager.load_json()
    
    # Add to vector store
    vs_manager.add_documents(documents)

    #Upload the chromadb to blob storage for version cotrol
    vs_manager.upload_chroma_to_blob()

    #breakpoint()
    
    # Test search
    results = vs_manager.similarity_search("What is my Python experience?")
    
    print("\n🔍 Search Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['relevance_score']:.4f}")
        print(f"   Source: {result['metadata'].get('source', 'Unknown')}")
        print(f"   Content: {result['content'][:200]}...")
