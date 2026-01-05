"""
OpenAI client wrapper for chat completions and embeddings
"""

import os
from typing import List, Dict, Any
from openai import OpenAI
from app.utils.logging import app_logger


class OpenAIClient:
    """Wrapper for OpenAI API operations"""
    
    def __init__(self, api_key: str, chat_model: str, embedding_model: str):
        """
        Initialize OpenAI client
        
        Args:
            api_key: OpenAI API key
            chat_model: Model name for chat completions
            embedding_model: Model name for embeddings
        """
        self.client = OpenAI(api_key=api_key)
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        app_logger.info(f"OpenAI client initialized with chat model: {chat_model}, embedding model: {embedding_model}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1000
    ) -> str:
        """
        Generate chat completion
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
        
        Returns:
            Generated response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            app_logger.error(f"Chat completion error: {str(e)}")
            raise
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for a single text
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector as list of floats
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            
            return response.data[0].embedding
        
        except Exception as e:
            app_logger.error(f"Embedding creation error: {str(e)}")
            raise
    
    def create_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Create embeddings for multiple texts in batches
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call
        
        Returns:
            List of embedding vectors
        """
        # Filter out empty strings and validate
        valid_texts = []
        valid_indices = []
        for idx, text in enumerate(texts):
            if text and isinstance(text, str) and text.strip():
                valid_texts.append(text.strip())
                valid_indices.append(idx)
            else:
                app_logger.warning(f"Skipping empty/invalid text at index {idx}")
        
        if not valid_texts:
            raise ValueError("No valid texts to embed")
        
        app_logger.info(f"Processing {len(valid_texts)} valid texts out of {len(texts)} total")
        
        all_embeddings = []
        
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                app_logger.info(f"Created embeddings for batch {i//batch_size + 1} ({len(batch)} texts)")
            
            except Exception as e:
                app_logger.error(f"Batch embedding error at index {i}: {str(e)}")
                app_logger.error(f"Problematic batch sample: {batch[0][:100] if batch else 'empty'}")
                raise
        
        return all_embeddings
