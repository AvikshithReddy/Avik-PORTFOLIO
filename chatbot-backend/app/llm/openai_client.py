"""
OpenAI + Cohere client wrappers for chat completions, embeddings, and rerank
"""

from typing import List, Dict, Any
from openai import OpenAI
import cohere
from app.utils.logging import app_logger


class OpenAIClient:
    """Wrapper for OpenAI API operations"""

    def __init__(self, api_key: str, chat_model: str, embedding_model: str):
        self.client = OpenAI(api_key=api_key)
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        app_logger.info(
            f"OpenAI client initialized with chat model: {chat_model}, embedding model: {embedding_model}"
        )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            app_logger.error(f"Chat completion error: {str(e)}")
            raise

    def create_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            app_logger.error(f"Embedding creation error: {str(e)}")
            raise

    def create_embeddings_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        valid_texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
        if not valid_texts:
            raise ValueError("No valid texts to embed")

        all_embeddings: List[List[float]] = []
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i : i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch,
                )
                all_embeddings.extend([item.embedding for item in response.data])
                app_logger.info(f"Embedded batch {i//batch_size + 1} ({len(batch)} texts)")
            except Exception as e:
                app_logger.error(f"Batch embedding error at index {i}: {str(e)}")
                raise

        return all_embeddings


class CohereReranker:
    """Thin wrapper around Cohere Rerank API"""

    def __init__(self, api_key: str, model: str = "rerank-3.5"):
        # Some environments may not supply a Cohere key; allow graceful fallback
        self.client = cohere.Client(api_key=api_key) if api_key else None
        self.model = model

    def rerank(self, query: str, documents: List[str], top_n: int = 8) -> List[int]:
        if not documents:
            return []

        if not self.client:
            # No Cohere client configured; return original ordering
            return list(range(min(top_n, len(documents))))

        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=[{"text": doc} for doc in documents],
            top_n=top_n,
        )

        # Return indices of reranked docs in descending order
        return [int(item.index) for item in response.results]
