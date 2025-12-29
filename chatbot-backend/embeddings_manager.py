"""
Embeddings Manager with Modern OpenAI API (v1.0+)
Implements efficient embedding generation, caching, and semantic retrieval
"""
import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from openai import OpenAI


class EmbeddingsManager:
    """Manages embeddings generation and caching using OpenAI API v1.0+"""
    
    def __init__(self, model: str = "text-embedding-3-small", cache_dir: str = ".embeddings_cache"):
        """
        Initialize embeddings manager
        
        Args:
            model: OpenAI embedding model (text-embedding-3-small or text-embedding-3-large)
            cache_dir: Directory for caching embeddings
        """
        self.model = model
        self.cache_dir = cache_dir
        self.cache = {}
        
        # Initialize OpenAI client safely
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = None
        
        if api_key:
            try:
                self.client = OpenAI(api_key=api_key)
                print("✅ OpenAI client initialized")
            except Exception as e:
                print(f"⚠️  OpenAI initialization error: {e}")
        else:
            print("⚠️  OPENAI_API_KEY not set - embeddings will use random fallback")
        
        # Create cache directory if needed
        os.makedirs(cache_dir, exist_ok=True)
        self._load_cache_from_disk()
    
    def get_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Get embedding for text with optional caching
        
        Args:
            text: Text to embed
            use_cache: Whether to use cache
            
        Returns:
            Embedding vector as numpy array
        """
        # Check cache first
        if use_cache and text in self.cache:
            return np.array(self.cache[text])
        
        # Return fallback if client not initialized
        if not self.client:
            return np.random.rand(1536)
        
        try:
            # Call OpenAI API v1.0+
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            
            embedding = np.array(response.data[0].embedding)
            
            # Cache the embedding
            if use_cache:
                self.cache[text] = embedding.tolist()
            
            return embedding
            
        except Exception as e:
            print(f"⚠️  Embedding error: {e}")
            # Fallback: random embedding (for development)
            return np.random.rand(1536)
    
    def get_embeddings_batch(self, texts: List[str], use_cache: bool = True) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts efficiently
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use cache
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        # Identify which texts need embedding
        for i, text in enumerate(texts):
            if use_cache and text in self.cache:
                embeddings.append(np.array(self.cache[text]))
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # Batch API call for uncached texts
        if texts_to_embed:
            if not self.client:
                # Return fallback embeddings
                for _ in texts_to_embed:
                    embeddings.append(np.random.rand(1536))
                return embeddings
            
            try:
                response = self.client.embeddings.create(
                    input=texts_to_embed,
                    model=self.model
                )
                
                for idx, data in enumerate(response.data):
                    embedding = np.array(data.embedding)
                    embeddings.insert(indices_to_embed[idx], embedding)
                    
                    # Cache
                    if use_cache:
                        self.cache[texts_to_embed[idx]] = embedding.tolist()
                        
            except Exception as e:
                print(f"⚠️  Batch embedding error: {e}")
                # Fallback
                for _ in texts_to_embed:
                    embeddings.insert(len(embeddings), np.random.rand(1536))
        
        return embeddings
    
    def save_cache_to_disk(self):
        """Save embedding cache to disk"""
        try:
            cache_file = os.path.join(self.cache_dir, "embeddings_cache.json")
            # Convert numpy arrays to lists for JSON serialization
            cache_serializable = {k: v if isinstance(v, list) else v.tolist() 
                                 for k, v in self.cache.items()}
            with open(cache_file, 'w') as f:
                json.dump(cache_serializable, f)
            print(f"✅ Cached {len(self.cache)} embeddings")
        except Exception as e:
            print(f"⚠️  Cache save error: {e}")
    
    def _load_cache_from_disk(self):
        """Load embedding cache from disk"""
        try:
            cache_file = os.path.join(self.cache_dir, "embeddings_cache.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
                print(f"✅ Loaded {len(self.cache)} cached embeddings")
        except Exception as e:
            print(f"⚠️  Cache load error: {e}")


class SemanticSearchEngine:
    """High-performance semantic search using embeddings"""
    
    def __init__(self, embeddings_manager: EmbeddingsManager):
        """
        Initialize search engine
        
        Args:
            embeddings_manager: Manager for generating embeddings
        """
        self.embeddings_manager = embeddings_manager
        self.documents = []
        self.embeddings_matrix = None
    
    def add_documents(self, documents: List[Dict]):
        """
        Add documents to index
        
        Args:
            documents: List of document dicts with 'text' and 'metadata' keys
        """
        self.documents.extend(documents)
    
    def build_index(self, batch_size: int = 100):
        """
        Build search index by generating embeddings for all documents
        
        Args:
            batch_size: Number of documents to process per batch
        """
        if not self.documents:
            print("⚠️  No documents to index")
            return
        
        print(f"🏗️  Building index for {len(self.documents)} documents...")
        
        # Extract texts
        texts = [doc.get('text', '') for doc in self.documents]
        
        # Get embeddings in batches
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embeddings_manager.get_embeddings_batch(batch_texts)
            embeddings.extend(batch_embeddings)
            
            if (i + batch_size) % (batch_size * 5) == 0:
                print(f"  ✓ Processed {min(i + batch_size, len(texts))} documents")
        
        # Build matrix
        self.embeddings_matrix = np.array(embeddings)
        
        # Store embeddings in documents
        for i, doc in enumerate(self.documents):
            doc['embedding'] = embeddings[i]
        
        print(f"✅ Index ready with {len(self.documents)} documents")
        
        # Save cache
        self.embeddings_manager.save_cache_to_disk()
    
    def search(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict]:
        """
        Semantic search for documents most relevant to query
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of most relevant documents with similarity scores
        """
        if self.embeddings_matrix is None:
            print("⚠️  Index not built. Call build_index() first.")
            return []
        
        # Get query embedding
        query_embedding = self.embeddings_manager.get_embedding(query)
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings_matrix)[0]
        
        # Get results above threshold
        results = []
        for idx, score in enumerate(similarities):
            if score >= threshold:
                doc = self.documents[idx].copy()
                doc['similarity_score'] = float(score)
                results.append(doc)
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]
    
    def search_by_type(self, query: str, doc_type: str, top_k: int = 3) -> List[Dict]:
        """
        Search for documents of specific type
        
        Args:
            query: Search query
            doc_type: Document type to filter (e.g., 'project', 'skill', 'github')
            top_k: Number of results to return
            
        Returns:
            Filtered results of specified type
        """
        all_results = self.search(query, top_k=top_k*3)
        filtered = [d for d in all_results if d.get('metadata', {}).get('type') == doc_type]
        return filtered[:top_k]
    
    def hybrid_search(self, query: str, keyword_words: Optional[List[str]] = None, 
                     top_k: int = 5) -> List[Dict]:
        """
        Hybrid search combining semantic and keyword matching
        
        Args:
            query: Search query
            keyword_words: Optional keywords for hybrid search
            top_k: Number of results
            
        Returns:
            Combined results from semantic + keyword search
        """
        # Semantic search
        semantic_results = self.search(query, top_k=top_k)
        
        # Keyword search (if keywords provided)
        keyword_results = []
        if keyword_words:
            query_lower = query.lower()
            for doc in self.documents:
                doc_text = (doc.get('text', '') + ' ' + 
                           str(doc.get('metadata', {}))).lower()
                if any(kw.lower() in doc_text for kw in keyword_words):
                    doc_copy = doc.copy()
                    doc_copy['similarity_score'] = 0.5  # Lower score for keyword match
                    keyword_results.append(doc_copy)
        
        # Combine and deduplicate
        combined = semantic_results + keyword_results
        seen_ids = set()
        unique_results = []
        
        for result in combined:
            chunk_id = result.get('chunk_id', '')
            if chunk_id not in seen_ids:
                unique_results.append(result)
                seen_ids.add(chunk_id)
        
        # Re-sort by similarity
        unique_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return unique_results[:top_k]


class VectorMemoryStore:
    """Simple in-memory vector store for conversation context"""
    
    def __init__(self, embeddings_manager: EmbeddingsManager, max_memories: int = 20):
        """
        Initialize memory store
        
        Args:
            embeddings_manager: Manager for embeddings
            max_memories: Maximum memories to store per session
        """
        self.embeddings_manager = embeddings_manager
        self.max_memories = max_memories
        self.session_memories = {}  # session_id -> list of memories
    
    def remember(self, session_id: str, text: str, metadata: Dict = None):
        """
        Remember a conversation turn
        
        Args:
            session_id: Session identifier
            text: Message text
            metadata: Optional metadata
        """
        if session_id not in self.session_memories:
            self.session_memories[session_id] = []
        
        memory = {
            'text': text,
            'embedding': self.embeddings_manager.get_embedding(text),
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.session_memories[session_id].append(memory)
        
        # Keep only recent memories
        if len(self.session_memories[session_id]) > self.max_memories:
            self.session_memories[session_id] = self.session_memories[session_id][-self.max_memories:]
    
    def recall_relevant(self, session_id: str, query: str, top_k: int = 3) -> List[Dict]:
        """
        Recall relevant memories for given query
        
        Args:
            session_id: Session identifier
            query: Query text
            top_k: Number of memories to return
            
        Returns:
            Most relevant memories
        """
        if session_id not in self.session_memories:
            return []
        
        query_embedding = self.embeddings_manager.get_embedding(query)
        memories = self.session_memories[session_id]
        
        # Score each memory
        scored_memories = []
        for memory in memories:
            similarity = cosine_similarity(
                [query_embedding],
                [memory['embedding']]
            )[0][0]
            scored_memories.append({
                **memory,
                'similarity_score': float(similarity)
            })
        
        # Sort and return top-k
        scored_memories.sort(key=lambda x: x['similarity_score'], reverse=True)
        return scored_memories[:top_k]
    
    def get_session_context(self, session_id: str) -> str:
        """Get full session context as string"""
        if session_id not in self.session_memories:
            return ""
        
        context_lines = []
        for memory in self.session_memories[session_id]:
            context_lines.append(memory['text'])
        
        return "\n".join(context_lines)
