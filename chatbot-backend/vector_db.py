"""
Vector Database Integration
Implements semantic search with embeddings and vector storage
"""
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime
import json
import os


class VectorStore:
    """
    Simple in-memory vector store for semantic search
    In production, use pgvector/PostgreSQL or Pinecone
    """
    
    def __init__(self, embedding_dim: int = 1536):
        """
        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}
        self.vector_index: List[str] = []  # Track insertion order
    
    def add(self,
            doc_id: str,
            embedding: np.ndarray,
            text: str,
            metadata: Dict = None) -> None:
        """Add vector to store"""
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: {embedding.shape[0]} vs {self.embedding_dim}")
        
        self.vectors[doc_id] = embedding
        self.metadata[doc_id] = {
            "text": text[:500],  # Store truncated text
            "timestamp": datetime.now().isoformat(),
            **(metadata or {})
        }
        if doc_id not in self.vector_index:
            self.vector_index.append(doc_id)
    
    def search(self,
               query_embedding: np.ndarray,
               top_k: int = 5,
               similarity_threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Search for similar vectors using cosine similarity
        
        Args:
            query_embedding: Query embedding vector
            top_k: Return top K results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of (doc_id, similarity_score) tuples
        """
        if not self.vectors:
            return []
        
        # Calculate cosine similarity
        scores = []
        for doc_id, vector in self.vectors.items():
            # Cosine similarity
            dot_product = np.dot(query_embedding, vector)
            norm_q = np.linalg.norm(query_embedding)
            norm_v = np.linalg.norm(vector)
            
            if norm_q > 0 and norm_v > 0:
                similarity = dot_product / (norm_q * norm_v)
            else:
                similarity = 0.0
            
            if similarity >= similarity_threshold:
                scores.append((doc_id, similarity))
        
        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    def get(self, doc_id: str) -> Optional[Dict]:
        """Get metadata for a document"""
        return self.metadata.get(doc_id)
    
    def delete(self, doc_id: str) -> bool:
        """Delete a document"""
        if doc_id in self.vectors:
            del self.vectors[doc_id]
            del self.metadata[doc_id]
            self.vector_index.remove(doc_id)
            return True
        return False
    
    def size(self) -> int:
        """Get number of vectors in store"""
        return len(self.vectors)
    
    def clear(self) -> None:
        """Clear all vectors"""
        self.vectors.clear()
        self.metadata.clear()
        self.vector_index.clear()
    
    def save(self, filepath: str) -> None:
        """Save vector store to disk"""
        try:
            data = {
                "embedding_dim": self.embedding_dim,
                "metadata": self.metadata,
                "vector_index": self.vector_index,
                # Note: numpy arrays are saved as lists
                "vectors": {doc_id: vec.tolist() for doc_id, vec in self.vectors.items()}
            }
            with open(filepath, 'w') as f:
                json.dump(data, f)
            print(f"✅ Vector store saved to {filepath}")
        except Exception as e:
            print(f"❌ Error saving vector store: {e}")
    
    def load(self, filepath: str) -> None:
        """Load vector store from disk"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.embedding_dim = data.get("embedding_dim", 1536)
            self.metadata = data.get("metadata", {})
            self.vector_index = data.get("vector_index", [])
            
            # Convert lists back to numpy arrays
            self.vectors = {
                doc_id: np.array(vec)
                for doc_id, vec in data.get("vectors", {}).items()
            }
            print(f"✅ Vector store loaded from {filepath}")
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")


class SemanticRetrieval:
    """
    Semantic retrieval system using embeddings
    Retrieves relevant documents based on query similarity
    """
    
    def __init__(self, vector_store: VectorStore, embeddings_manager):
        """
        Args:
            vector_store: VectorStore instance
            embeddings_manager: EmbeddingsManager for generating embeddings
        """
        self.vector_store = vector_store
        self.embeddings_manager = embeddings_manager
    
    def retrieve(self,
                 query: str,
                 top_k: int = 5,
                 similarity_threshold: float = 0.5) -> List[Dict]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query text
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of retrieved documents with scores
        """
        # Get query embedding
        query_embedding = self.embeddings_manager.get_embedding(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        # Format results
        retrieved = []
        for doc_id, score in results:
            metadata = self.vector_store.get(doc_id)
            if metadata:
                retrieved.append({
                    "doc_id": doc_id,
                    "text": metadata.get("text", ""),
                    "score": float(score),
                    "source": metadata.get("source", "unknown"),
                    "metadata": {k: v for k, v in metadata.items() if k not in ["text"]}
                })
        
        return retrieved
    
    def retrieve_by_source(self,
                          query: str,
                          source_type: str,
                          top_k: int = 5) -> List[Dict]:
        """Retrieve documents from a specific source"""
        all_results = self.retrieve(query, top_k=top_k*2)
        
        # Filter by source
        filtered = [r for r in all_results if r.get("source") == source_type]
        
        return filtered[:top_k]


class DocumentIndexer:
    """
    Indexes documents into vector store
    Handles chunking and embedding
    """
    
    def __init__(self,
                 vector_store: VectorStore,
                 embeddings_manager,
                 chunk_size: int = 500,
                 chunk_overlap: int = 100):
        """
        Args:
            vector_store: VectorStore instance
            embeddings_manager: EmbeddingsManager
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.vector_store = vector_store
        self.embeddings_manager = embeddings_manager
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def index_document(self,
                       doc_id: str,
                       text: str,
                       source: str,
                       metadata: Dict = None) -> int:
        """
        Index a document by chunking and embedding
        
        Args:
            doc_id: Document ID
            text: Document text
            source: Source type (e.g., "resume", "github")
            metadata: Additional metadata
            
        Returns:
            Number of chunks indexed
        """
        # Chunk text
        chunks = self._chunk_text(text)
        
        # Embed and store chunks
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            
            # Get embedding
            embedding = self.embeddings_manager.get_embedding(chunk)
            
            # Add to vector store
            chunk_metadata = {
                "source": source,
                "original_doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(metadata or {})
            }
            
            self.vector_store.add(chunk_id, embedding, chunk, chunk_metadata)
        
        return len(chunks)
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for period, newline, or other break
                for marker in ['. ', '\n\n', '\n', ', ']:
                    break_pos = text.rfind(marker, start, end)
                    if break_pos > start:
                        end = break_pos + len(marker)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start with overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def index_multiple_documents(self, documents: List[Dict]) -> int:
        """
        Index multiple documents
        
        Args:
            documents: List of dicts with 'id', 'text', 'source', and optional 'metadata'
            
        Returns:
            Total number of chunks indexed
        """
        total_chunks = 0
        
        for doc in documents:
            chunks = self.index_document(
                doc_id=doc.get("id", f"doc_{len(documents)}"),
                text=doc.get("text", ""),
                source=doc.get("source", "unknown"),
                metadata=doc.get("metadata", {})
            )
            total_chunks += chunks
            print(f"✅ Indexed {doc.get('source', 'document')}: {chunks} chunks")
        
        return total_chunks
