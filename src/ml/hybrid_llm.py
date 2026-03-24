import logging
from typing import Literal

logger = logging.getLogger(__name__)

class HybridLLM:
    """Wrapper that tries Featherless first, falls back to Ollama"""
    
    def __init__(
        self, 
        featherless_llm,  # Existing HuggingFaceInferenceLLM instance
        ollama_llm,       # New OllamaLLM instance
        strategy: Literal["hybrid", "featherless-only", "ollama-only"] = "hybrid"
    ):
        self.featherless = featherless_llm
        self.ollama = ollama_llm
        self.strategy = strategy
        self._featherless_failures = 0
        self._max_failures = 1
    
    def invoke(self, input_data, **kwargs):
        """Invoke with fallback logic"""
        
        # Force featherless only
        if self.strategy == "featherless-only":
            logger.info("Using Featherless (forced)")
            return self.featherless.invoke(input_data, **kwargs)
        
        # Force ollama only
        if self.strategy == "ollama-only":
            logger.info("Using Ollama (forced)")
            return self.ollama.invoke(input_data, **kwargs)
        
        # Hybrid: try featherless first, fallback to ollama
        try:
            # Circuit breaker check
            if self._featherless_failures >= self._max_failures:
                logger.warning("Featherless circuit open, using Ollama")
                raise Exception("Circuit breaker open")
            
            logger.info("Attempting Featherless (primary)")
            response = self.featherless.invoke(input_data, **kwargs)
            self._featherless_failures = 0  # Reset on success
            logger.info("✅ Featherless succeeded")
            return response
        
        except Exception as e:
            self._featherless_failures += 1
            logger.warning(f"Featherless failed (attempt {self._featherless_failures}): {e}")
            logger.info("🔄 Falling back to Ollama")
            
            # Fallback
            try:
                response = self.ollama.invoke(input_data, **kwargs)
                logger.info("✅ Ollama succeeded")
                return response
            except Exception as ollama_error:
                logger.error(f"❌ Both providers failed")
                raise Exception(f"All LLM providers failed. Featherless: {e}, Ollama: {ollama_error}")
            
    def __call__(self, prompt, **kwargs):
        """Make the class callable"""
        return self.invoke(prompt, **kwargs)