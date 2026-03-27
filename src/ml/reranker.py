import logging
import requests
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class JinaReranker:
    """Reranker using Jina AI API"""
    def __init__(self, api_key: str, model: str = "jina-reranker-v2-base-multilingual"):
        self.api_url = "https://api.jina.ai/v1/rerank"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.model = model
        logger.info(f"✅ Jina Reranker initialized: {model}")

    def rerank(self, query: str, documents: List[str], top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Rerank documents using Jina AI API
        """
        if not documents:
            return []

        try:
            payload = {
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": top_k
            }

            logger.debug(f"🔄 Reranking {len(documents)} documents with Jina AI")

            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )

            if response.status_code != 200:
                logger.warning(f"⚠️ Jina reranking failed: {response.status_code} - {response.text}")
                return []  # Return empty, no reranking

            result = response.json()

            # Jina returns: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
            if "results" in result:
                top_results = [(item['index'], item['relevance_score']) for item in result['results']]
                logger.info(f"✅ Jina reranked to top {len(top_results)} documents")
                logger.debug(f"Top scores: {[f'{score:.3f}' for _, score in top_results[:3]]}")
                return top_results
            else:
                logger.warning(f"⚠️ Unexpected Jina response format: {result}")
                return []  # Return empty, no reranking

        except Exception as e:
            logger.error(f"❌ Jina reranking error: {e}")
            return []  # Return empty, no reranking
