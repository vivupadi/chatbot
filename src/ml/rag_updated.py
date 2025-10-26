from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

import time
import logging
from typing import Any, List, Optional
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from huggingface_hub import InferenceClient
from langchain_community.llms import Ollama

import os
import requests



from src.data.vectorstore import VectorStore
from src.utils.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HuggingFaceInferenceLLM:  # LangChain-compatible wrapper for HuggingFace InferenceClient
    def __init__(self, api_key: str, model: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.95):
        # Use Featherless AI directly
        self.api_url = "https://api.featherless.ai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        logger.info(f"✅ Using Featherless AI: {model}")
    
    def invoke(self, input_data, **kwargs):
        """Invoke the model with a prompt"""
        # Extract prompt text
        if hasattr(input_data, 'to_string'):
            prompt = input_data.to_string()
        else:
            prompt = str(input_data)
        
        # Prepare the payload
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "temperature": kwargs.get('temperature', self.temperature),
            "top_p": kwargs.get('top_p', self.top_p),
        }
        
        try:
            logger.debug(f"🔄 Calling Featherless AI with model: {self.model}")
            response = requests.post(
                self.api_url, 
                headers=self.headers, 
                json=payload, 
                timeout=60
            )
            
            # Better error handling
            if response.status_code == 401:
                logger.error("❌ Featherless AI authentication failed!")
                logger.error(f"Check your API key starts correctly")
                logger.error(f"Response: {response.text}")
            
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Featherless AI call failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise
    
    def __call__(self, prompt, **kwargs):
        """Make the class callable"""
        return self.invoke(prompt, **kwargs)
    


class RAGengine:
    def __init__(self, vectorstore=None):
        logger.info("Initializing RAG Engine")

        # Initialize vectorstore
        if vectorstore is not None:
            self.vector_store = vectorstore
        else:
            self.vector_store = VectorStore()

        # Initialize LLM
        self._init_llm()

        # Setup RAG chain
        self._setup_chain()

        logger.info("RAG Engine Initialized")
    
    def _init_llm(self):
        if settings.use_ollama:
            try:

                print("✅ Using Ollama")
                self.llm = Ollama(
                    base_url=settings.ollama_base_url,
                    model=settings.ollama_model,
                    temperature=0.7
                )
                logger.info(f"Using Ollama: {settings.ollama_model}")

            except Exception as e:
                logger.warning(f"Ollama not available: {e}")
                logger.info("Falling back to HuggingFace...")
                self._init_huggingface_llm()
        else:
            self._init_huggingface_llm()

    def _init_huggingface_llm(self):
        # Get Featherless API key (NOT HuggingFace token)
        api_key = settings.featherless_ai_key_new or os.environ.get("FEATHERLESS_API_KEY")
        
        if not api_key:
            raise ValueError("FEATHERLESS_API_KEY required! Get it from https://featherless.ai/")
        
        # Log partial key for debugging
        logger.info(f"🔑 Using Featherless key: {api_key[:8]}...{api_key[-4:]}")
        
        model = getattr(settings, 'huggingface_model', 'mistralai/Mistral-7B-Instruct-v0.2')
        
        self.llm = HuggingFaceInferenceLLM(api_key=api_key, model=model)
        logger.info(f"✅ LLM initialized")

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
        print(f"✅ Prompt created")

        # Get retriever from vector store
        retriever = self.vector_store.get_retriever()
        print(f"✅ Retriever created")

        # Create RAG chain using LCEL (LangChain Expression Language)
        # CORRECT LCEL CHAIN STRUCTURE
        from langchain_core.runnables import RunnableParallel, RunnableLambda
        
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
    

    llm = RAGengine()
    sucess = llm._init_huggingface_llm

    #print(f"model successfully initiaized:{sucess}++++++")
    #breakpoint()
    # Test queries
    test_questions = [
        "What is my Python experience?",
        "Tell me about my projects",
        "What are my technical skills?",
        "What is my educational background?",
        "What is my Job experience?"
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

        