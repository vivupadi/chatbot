#LLM class
import time
import logging
from typing import Any, List, Optional

import requests


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
    