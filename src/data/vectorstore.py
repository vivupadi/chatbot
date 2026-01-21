import chromadb
from chromadb.config import Settings as ChromaSettings

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional
import logging
from pathlib import Path
from src.utils.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        logger.info("Initialize Embeddings..")
        self.embeddings = HuggingFaceEmbeddings(
            model_name = settings.embedding_model,
            model_kwargs = {'device': settings.embedding_device},
            encode_kwargs={'normalize_embeddings': True}
        )

        logger.info("🔧 Initializing ChromaDB...")
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
        path=str(settings.data_vector_dir),
        settings=ChromaSettings(anonymized_telemetry=False)  # Disables telemetry
        )

        # Initialize vector store
        self.vector_store = Chroma(
            client=self.client,
            collection_name="rag_documents",
            embedding_function=self.embeddings
        )

        logger.info("Vector store initialized")

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

    def upload_chromadb_to_blob(self, local_chroma_path, CONNECT_STR, CONTAINER_NAME):
        """Upload entire ChromaDB folder to blob"""
        print("Uploading ChromaDB to blob...")
        blob_service = BlobServiceClient.from_connection_string(CONNECT_STR)
        container = blob_service.get_container_client(CONTAINER_NAME)
        
        # Upload all ChromaDB files
        for file in Path(local_chroma_path).rglob("*"):
            if file.is_file():
                blob_name = f"chromadb/{file.relative_to(local_chroma_path)}"
                with open(file, 'rb') as data:
                    container.get_blob_client(blob_name).upload_blob(data, overwrite=True)
        
        print("✓ Uploaded ChromaDB to blob")
    
    def download_chromadb_from_blob(self, CONNECT_STR, CONTAINER_NAME):
        """Download ChromaDB from blob to Hetzner"""
        print("Downloading ChromaDB from blob to production...")
        blob_service = BlobServiceClient.from_connection_string(CONNECT_STR)
        container = blob_service.get_container_client(CONTAINER_NAME)
        
        # Clear existing ChromaDB
        if Path(CHROMA_PATH).exists():
            shutil.rmtree(CHROMA_PATH)
        Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        
        # Download all ChromaDB files
        for blob in container.list_blobs(name_starts_with="chromadb/"):
            local_file = Path(CHROMA_PATH) / blob.name.replace("chromadb/", "")
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(local_file, 'wb') as f:
                f.write(container.get_blob_client(blob.name).download_blob().readall())
        
        print("✓ Downloaded ChromaDB to production")



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


# Example usage
if __name__ == "__main__":
    import json
    
    # Initialize vector store
    vs_manager = VectorStore()
    
    # Load documents
    with open("./data/processed/documents.json", 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Add to vector store
    vs_manager.add_documents(documents)
    
    # Test search
    results = vs_manager.similarity_search("What is my Python experience?")
    
    print("\n🔍 Search Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['relevance_score']:.4f}")
        print(f"   Source: {result['metadata'].get('source', 'Unknown')}")
        print(f"   Content: {result['content'][:200]}...")
