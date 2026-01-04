"""
RAG Engine with Persistent Embeddings & Fast Vector Search
Optimized for quick response times with pre-computed embeddings
"""
import os
import json
import pickle
import hashlib
import numpy as np
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from datetime import datetime


class RAGEngine:
    """
    Retrieval-Augmented Generation Engine with:
    - Persistent embedding storage (no re-computation)
    - Fast numpy-based vector similarity search
    - Smart caching for GitHub and portfolio data
    - Optimized top-k retrieval
    """
    
    def __init__(self, openai_api_key: str, cache_dir: str = "./embeddings_cache"):
        """Initialize RAG engine with caching"""
        self.client = OpenAI(api_key=openai_api_key)
        self.cache_dir = cache_dir
        self.embeddings_file = os.path.join(cache_dir, "document_embeddings.pkl")
        self.metadata_file = os.path.join(cache_dir, "metadata.json")
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Storage for pre-computed embeddings
        self.document_embeddings = []  # List of numpy arrays
        self.documents = []  # List of document dicts
        self.embedding_dim = 1536  # text-embedding-3-small dimension
        
        # Load cached embeddings if available
        self._load_cache()
    
    def _compute_text_hash(self, text: str) -> str:
        """Compute hash for text to check if it's changed"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Get embedding for text with optional caching
        Returns: numpy array of shape (1536,)
        """
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return embedding
        except Exception as e:
            print(f"⚠️  Embedding error: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def batch_get_embeddings(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts in batches
        Much faster than individual calls
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
                all_embeddings.extend(embeddings)
                print(f"  Embedded batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            except Exception as e:
                print(f"⚠️  Batch embedding error: {e}")
                # Add zero vectors as fallback
                all_embeddings.extend([np.zeros(self.embedding_dim, dtype=np.float32)] * len(batch))
        
        return all_embeddings
    
    def cosine_similarity_batch(self, query_embedding: np.ndarray, 
                               document_embeddings: np.ndarray) -> np.ndarray:
        """
        Fast vectorized cosine similarity computation
        query_embedding: shape (1536,)
        document_embeddings: shape (n_docs, 1536)
        Returns: shape (n_docs,) with similarity scores
        """
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        # Normalize documents (along dimension 1)
        doc_norms = np.linalg.norm(document_embeddings, axis=1, keepdims=True) + 1e-10
        docs_normalized = document_embeddings / doc_norms
        
        # Compute dot product (cosine similarity)
        similarities = np.dot(docs_normalized, query_norm)
        
        return similarities
    
    def add_documents(self, documents: List[Dict], force_recompute: bool = False):
        """
        Add documents to the RAG system with embeddings
        Each document should have: {"content": str, "type": str, "metadata": dict}
        """
        if not force_recompute and self._is_cache_valid(documents):
            print("✅ Using cached embeddings (no changes detected)")
            return
        
        print(f"🔄 Computing embeddings for {len(documents)} documents...")
        
        # Extract texts for batch embedding
        texts = [doc["content"] for doc in documents]
        
        # Batch compute embeddings (much faster!)
        embeddings = self.batch_get_embeddings(texts, batch_size=50)
        
        # Store as numpy array for fast operations
        self.document_embeddings = np.array(embeddings, dtype=np.float32)
        self.documents = documents
        
        # Save to cache
        self._save_cache()
        
        print(f"✅ Embedded {len(documents)} documents")
        print(f"   Embedding matrix shape: {self.document_embeddings.shape}")
    
    def retrieve_top_k(self, query: str, top_k: int = 8, 
                       min_similarity: float = 0.25) -> List[Tuple[float, Dict]]:
        """
        Retrieve top-k most relevant documents using fast vector search
        Returns: List of (similarity_score, document) tuples
        """
        if len(self.documents) == 0:
            return []
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            return []
        
        # Fast batch cosine similarity
        similarities = self.cosine_similarity_batch(query_embedding, self.document_embeddings)
        
        # Apply keyword boosting
        query_lower = query.lower()
        keyword_boosts = np.zeros(len(self.documents), dtype=np.float32)
        
        for idx, doc in enumerate(self.documents):
            doc_type = doc.get("type", "")
            boost = 0.0
            
            if "experience" in query_lower and "experience" in doc_type:
                boost = 0.3
            elif "project" in query_lower and "project" in doc_type:
                boost = 0.3
            elif "skill" in query_lower and "skill" in doc_type:
                boost = 0.3
            elif "github" in query_lower and doc_type == "github":
                boost = 0.2
            elif "education" in query_lower and doc_type == "education":
                boost = 0.3
            
            keyword_boosts[idx] = boost
        
        # Combined scores
        combined_scores = similarities + keyword_boosts
        
        # Filter by minimum similarity
        valid_indices = np.where(similarities >= min_similarity)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # Get top-k indices
        valid_scores = combined_scores[valid_indices]
        top_indices_in_valid = np.argsort(valid_scores)[::-1][:top_k]
        top_indices = valid_indices[top_indices_in_valid]
        
        # Build results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            results.append((score, self.documents[idx]))
        
        return results
    
    def format_context(self, retrieved_docs: List[Tuple[float, Dict]]) -> str:
        """Format retrieved documents into context string for LLM"""
        if not retrieved_docs:
            return ""
        
        context = "📋 RELEVANT CONTEXT FROM PORTFOLIO:\n\n"
        
        for score, doc in retrieved_docs:
            doc_type = doc.get('type', 'UNKNOWN').upper()
            content = doc.get('content', '')
            context += f"[{doc_type}] (Relevance: {score:.2f})\n{content}\n\n"
        
        return context
    
    def _save_cache(self):
        """Save embeddings and documents to disk"""
        try:
            # Save embeddings as numpy array
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump({
                    'embeddings': self.document_embeddings,
                    'documents': self.documents
                }, f)
            
            # Save metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'num_documents': len(self.documents),
                'embedding_dim': self.embedding_dim,
                'doc_hashes': [self._compute_text_hash(doc['content']) for doc in self.documents]
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            print(f"💾 Cached {len(self.documents)} embeddings to disk")
        except Exception as e:
            print(f"⚠️  Cache save error: {e}")
    
    def _load_cache(self):
        """Load cached embeddings from disk"""
        try:
            if not os.path.exists(self.embeddings_file):
                print("📂 No embedding cache found (will create on first run)")
                return
            
            with open(self.embeddings_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.document_embeddings = cache_data['embeddings']
            self.documents = cache_data['documents']
            
            print(f"✅ Loaded {len(self.documents)} cached embeddings")
            print(f"   Cache shape: {self.document_embeddings.shape}")
        except Exception as e:
            print(f"⚠️  Cache load error: {e}")
            self.document_embeddings = []
            self.documents = []
    
    def _is_cache_valid(self, new_documents: List[Dict]) -> bool:
        """Check if cache is still valid (documents haven't changed)"""
        if not os.path.exists(self.metadata_file):
            return False
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check document count
            if metadata['num_documents'] != len(new_documents):
                return False
            
            # Check content hashes
            cached_hashes = metadata.get('doc_hashes', [])
            new_hashes = [self._compute_text_hash(doc['content']) for doc in new_documents]
            
            return cached_hashes == new_hashes
        except Exception as e:
            print(f"⚠️  Cache validation error: {e}")
            return False
    
    def clear_cache(self):
        """Clear all cached embeddings"""
        if os.path.exists(self.embeddings_file):
            os.remove(self.embeddings_file)
        if os.path.exists(self.metadata_file):
            os.remove(self.metadata_file)
        print("🗑️  Cache cleared")


# Quick test
if __name__ == "__main__":
    # Test the RAG engine
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "")
    
    if api_key:
        rag = RAGEngine(api_key)
        
        # Test documents
        test_docs = [
            {"content": "Python expert with 5 years experience", "type": "skill", "metadata": {}},
            {"content": "Built ML pipeline with PyTorch", "type": "project", "metadata": {}},
        ]
        
        rag.add_documents(test_docs)
        results = rag.retrieve_top_k("What Python experience do you have?", top_k=2)
        
        print("\n🔍 Test Query Results:")
        for score, doc in results:
            print(f"  Score: {score:.3f} | {doc['content'][:50]}...")
    else:
        print("⚠️  Set OPENAI_API_KEY to test")
