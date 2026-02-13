import httpx
import logging

logger = logging.getLogger(__name__)

class OllamaLLM:
    """Ollama provider matching HuggingFaceInferenceLLM interface"""
    
    def __init__(self, base_url: str = "http://ollama-service:11434", 
                 model: str = "phi", max_tokens: int = 512, 
                 temperature: float = 0.7):
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = httpx.Client(timeout=120.0)
        logger.info(f"✅ Ollama initialized: {model}")
    
    def invoke(self, input_data, **kwargs):
        """Invoke method matching existing interface"""
        # Extract prompt text (same as HuggingFaceInferenceLLM)
        if hasattr(input_data, 'to_string'):
            prompt = input_data.to_string()
        else:
            prompt = str(input_data)
        
        # Prepare payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": kwargs.get('max_tokens', self.max_tokens),
                "temperature": kwargs.get('temperature', self.temperature)
            }
        }
        
        try:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data["response"]
        
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise

    def __call__(self, prompt, **kwargs):
        """Make the class callable"""
        return self.invoke(prompt, **kwargs)