from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Optional
import logging
import time

from src.data.vectorstore import VectorStore
from src.utils.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGengine:
    def __init__(self, vectorstore = None):
        logger.info("Initializing RAG Engine")

        #Initialize vectorstore
        self.vector_store = VectorStore()

        #Initialize LLM
        self._init_llm()


        #Setup RAG chain
        self._setup_chain()

        logger.info("RAG Engine Initialized")

    def _init_llm(self):
        if settings.use_ollama:
            try:
                from langchain_community.llms import Ollama
                self.llm = Ollama(
                    base_url = settings.ollama_base_url,
                    model = settings.ollama_model,
                    temperature= 0.7
                )
                logger.info(f"Using Ollama:{settings.ollama_model}")
            except Exception as e:
                logger.warning(f"Ollama not available:{e}")
                logger.info("Falling back to HuggingFAce...")
                self._init_huggingface_llm()
        else:
            self._init_huggingface_llm()

    def _init_huggingface_llm(self):
        if not settings.huggingface_token:
            logger.warning("No HuggingFace Token found")

        self.llm = HuggingFaceEndpoint(
            repo_id = settings.huggingface_model,
            huggingfacehub_api_token=settings.huggingface_token,
            #max_new_token = 512,
            temperature= 0.7,
            top_p = 0.95
        )
        logger.info(f"✅ Using Hugging Face: {settings.huggingface_model}")

    def _format_docs(self, docs):
        """Format retrieved documents for context"""
        return "\n\n".join([doc.page_content for doc in docs])

    def _setup_chain(self):
        template = """You are a helpful AI assistant. Use the following context to answer the question.
        If you don't know the answer, just say you don't know. Don't make up information.

        Context:
        {context}

        Question: {question}

        Answer (be concise and accurate):"""

        prompt = PromptTemplate.from_template(template)

        # Get retriever from vector store
        retriever = self.vector_store.get_retriever()

        # Create RAG chain using LCEL (LangChain Expression Language)
        # This uses the pipe operator | to chain operations
        self.rag_chain = (
            {
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Store retriever for getting source documents separately
        self.retriever = retriever

        logger.info("✅ RAG chain configured")

    def query(self, question):

        logger.info(f"Query: {question}")
        start_time = time.time()

        try:
            # Get source documents first (for citation)
            source_docs = self.retriever.invoke(question)

            # Generate answer using RAG chain
            answer = self.rag_chain.invoke(question)

            #Calculate latency
            latency = time.time() - start_time

            # Format sources
            sources = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in source_docs
            ]

           # Format response
            response = {
                "question": question,
                "answer": answer.strip(),
                "sources": sources,
                "latency_seconds": round(latency, 2)
            }
            
            logger.info(f"✅ Answer generated in {latency:.2f}s")
            return response
        
        except Exception as e:
            logger.error(f"Error during query{e}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "sources": [],
                "latency_seconds": round(time.time() - start_time, 2)
            }


# Example usage
if __name__ == "__main__":
    # Initialize RAG engine
    rag = RAGengine()
    
    # Test queries
    test_questions = [
        "What is my Python experience?",
        "Tell me about my projects",
        "What are my technical skills?",
        "What is my educational background?"
    ]
    
    print("\n" + "="*60)
    print("🤖 RAG ENGINE TEST")
    print("="*60)
    
    for question in test_questions:
        print(f"\n❓ {question}")
        print("-" * 60)
        
        response = rag.query(question)
        
        print(f"💡 Answer: {response['answer']}")
        print(f"⏱️  Latency: {response['latency_seconds']}s")
        print(f"📚 Sources: {len(response['sources'])} documents")
        
        if response['sources']:
            print("\n📄 Top Source:")
            top_source = response['sources'][0]
            print(f"   {top_source['content'][:150]}...")

        