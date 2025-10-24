from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Dict,Any, List, Optional
import logging
import time

from langchain_core.language_models.llms import LLM

# New imports for HuggingFace
from huggingface_hub import InferenceClient

from src.data.vectorstore import VectorStore
from src.utils.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HuggingFaceInferenceLLM(LLM):  # LangChain-compatible wrapper for HuggingFace InferenceClient
    client: Any = None
    model: str = ""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95

    def __init__(
        self, 
        token: str, 
        model: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95
    ):
        super().__init__()
        self.client = InferenceClient(token=token)
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Call the HuggingFace InferenceClient"""
        try:
            response = self.client.text_generation(
                prompt,
                model=self.model,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stop_sequences=stop,
                **kwargs
            )
            return response
        except Exception as e:
            logger.error(f"HuggingFace inference error: {e}")
            raise
        


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

        try:
            self.llm = HuggingFaceInferenceLLM(
                token=settings.huggingface_token,
                model=settings.huggingface_model,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95
            )
            logger.info(f"✅ Using Hugging Face: {settings.huggingface_model}")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace InferenceClient: {e}")
            raise

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
            print(f"✅ source test: {len(source_docs)} docs found")

            #breakpoint()
            # Generate answer using RAG chain
            answer = self.rag_chain.invoke(question)
            print("✅ ✅ ✅ ✅ ✅ ")
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
            breakpoint()
            
            logger.info(f"✅ Answer generated in {latency:.2f}s")
            return response
        
        except Exception as e:
            logger.error(f"Error during query{e}")
            # ADD THESE LINES TO SEE THE REAL ERROR:
            import traceback
            print("\n" + "="*60)
            print("FULL ERROR DETAILS:")
            print("="*60)
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Error repr: {repr(e)}")
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

        