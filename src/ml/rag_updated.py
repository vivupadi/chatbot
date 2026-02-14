#from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

import time
import logging
from typing import Any, List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_community.llms import Ollama

import os
from dotenv import load_dotenv
load_dotenv(override=False)

from src.data.vectorstore import VectorStore
from src.utils.config import settings
from src.ml.llm_model import HuggingFaceInferenceLLM
from src.ml.ollama import OllamaLLM
from src.ml.hybrid_llm import HybridLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGengine:
    def __init__(self, vectorstore=None):
        logger.info("Initializing RAG Engine")

        # Initialize vectorstore
        if vectorstore is not None:
            self.vector_store = vectorstore
        else:
            self.vector_store = VectorStore()

        # Initialize LLM
        self._init_featherless_llm()  #init featherless api
        self._init_ollama()           #init ollama api
        self._init_hybrid_llm()       #init hybrid llm with the above two models and the strategy defined in config

        # Setup RAG chain
        self._setup_chain()

        logger.info("RAG Engine Initialized")
    
    def _init_ollama(self):
        self.ollama_llm = OllamaLLM()
        logger.info(f"✅ Ollama initialized")

    def _init_featherless_llm(self):
        # Get Featherless API key (NOT HuggingFace token)

        api_key = settings.Featherless_ai_key_new or os.getenv("Featherless_ai_key_new")
        
        if not api_key:
            raise ValueError("FEATHERLESS_API_KEY required! Get it from https://featherless.ai/")
        
        # Log partial key for debugging
        logger.info(f"🔑 Using Featherless key: {api_key[:8]}...{api_key[-4:]}")
        
        model = getattr(settings, 'huggingface_model', 'mistralai/Mistral-7B-Instruct-v0.2')
        
        self.featherless_llm = HuggingFaceInferenceLLM(api_key=api_key, model=model)
        logger.info(f"✅ LLM initialized")

    def _init_hybrid_llm(self):
        
        self.strategy = settings.LLM_STRATEGY
        self.llm = HybridLLM(self.featherless_llm, self.ollama_llm, self.strategy)
        logger.info(f"✅ Hybrid LLM initialized with strategy: {self.strategy}")

    def _format_docs(self, docs):
        """Format retrieved documents for context"""
        return "\n\n".join([doc.page_content for doc in docs])

    def _setup_chain(self):
        template = """You are a helpful AI assistant. Use the following context to answer the questions about Vivek.
        If you don't know the answer, just say you don't know. Don't make up information.

        Context:
        {context}

        Question: {question}

        Answer (be concise and accurate):"""

        prompt = PromptTemplate.from_template(template)
        print(f"✅ Prompt created")

        # Get retriever from vector store
        retriever = self.vector_store.get_retriever()
        print(f"✅ Retriever created")

        # Create RAG chain using LCEL (LangChain Expression Language)
        # CORRECT LCEL CHAIN STRUCTURE
        
        self.rag_chain = (
            RunnableParallel({
                "context": retriever | RunnableLambda(self._format_docs),
                "question": RunnableLambda(lambda x: x)
            })
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
        #breakpoint()

        try:
            # Get source documents first (for citation)
            source_docs = self.retriever.invoke(question)

            #breakpoint()
            # Generate answer using RAG chain
            answer = self.rag_chain.invoke(question)
            
            #Calculate latency
            latency = time.time() - start_time

            #breakpoint()
            #Format sources
            sources = [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in source_docs
            ]

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
            # ADD THESE LINES TO SEE THE REAL ERROR:
            import traceback
            #print("\n" + "="*60)
            print("FULL ERROR DETAILS:")
            print("="*60)
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            #print(f"Error repr: {repr(e)}")
            traceback.print_exc()
            print("="*60 + "\n")
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
    featherless_llm = rag._init_featherless_llm()
    ollama_llm = rag._init_ollama()

    llm = rag._init_hybrid_llm()   

    print(f"model successfully initiaized:{llm}++++++")
    #breakpoint()
    # Test queries
    test_questions = [
        "What is my address?",
        "Tell me about my projects",
        "What are my email address?",
        "What is my educational background?",
        "What is my Job experience?",
        "Does Vivek have MLOps experience??"
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

        