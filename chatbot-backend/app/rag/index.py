"""
RAG index management - embeddings storage and retrieval
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from app.utils.logging import app_logger


class RAGIndex:
    """Manages document embeddings and retrieval"""
    
    def __init__(self, index_dir: str):
        """
        Initialize RAG index
        
        Args:
            index_dir: Directory to store index files
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.embeddings_file = self.index_dir / "embeddings.npy"
        self.metadata_file = self.index_dir / "metadata.json"
        
        self.embeddings = None
        self.metadata = []
        self.loaded = False
    
    def save_index(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Save embeddings and metadata to disk
        
        Args:
            embeddings: Numpy array of embeddings (N x D)
            metadata: List of metadata dictionaries (length N)
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata must have same length")
        
        # Save embeddings as numpy array
        np.save(self.embeddings_file, embeddings)
        
        # Save metadata as JSON
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.embeddings = embeddings
        self.metadata = metadata
        self.loaded = True
        
        app_logger.info(f"Saved RAG index: {len(embeddings)} embeddings to {self.index_dir}")
    
    def load_index(self) -> bool:
        """
        Load embeddings and metadata from disk
        
        Returns:
            True if successfully loaded, False otherwise
        """
        if not self.embeddings_file.exists() or not self.metadata_file.exists():
            app_logger.warning("RAG index files not found")
            return False
        
        try:
            # Load embeddings
            self.embeddings = np.load(self.embeddings_file)
            
            # Load metadata
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            self.loaded = True
            app_logger.info(f"Loaded RAG index: {len(self.embeddings)} embeddings from {self.index_dir}")
            return True
        
        except Exception as e:
            app_logger.error(f"Error loading RAG index: {str(e)}")
            return False
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 6,
        filter_fn: callable = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for most similar documents using cosine similarity
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_fn: Optional function to filter results
        
        Returns:
            List of (metadata, score) tuples sorted by relevance
        """
        if not self.loaded or self.embeddings is None:
            app_logger.warning("RAG index not loaded")
            return []
        
        # Convert query to numpy array
        query_vec = np.array(query_embedding).reshape(1, -1)
        
        # Compute cosine similarity
        # cosine_sim = (A · B) / (||A|| * ||B||)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized_embeddings = self.embeddings / (norms + 1e-9)
        
        query_norm = np.linalg.norm(query_vec)
        normalized_query = query_vec / (query_norm + 1e-9)
        
        similarities = np.dot(normalized_embeddings, normalized_query.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more than needed for filtering
        
        # Build results with filtering
        results = []
        for idx in top_indices:
            if len(results) >= top_k:
                break
            
            metadata = self.metadata[idx]
            score = float(similarities[idx])
            
            # Apply filter if provided
            if filter_fn is None or filter_fn(metadata):
                results.append((metadata, score))
        
        app_logger.info(f"Search returned {len(results)} results (top score: {results[0][1]:.3f if results else 0})")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        if not self.loaded:
            return {"loaded": False}
        
        source_types = {}
        for meta in self.metadata:
            source_type = meta.get('source_type', 'unknown')
            source_types[source_type] = source_types.get(source_type, 0) + 1
        
        return {
            "loaded": True,
            "total_chunks": len(self.metadata),
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "source_types": source_types
        }
    
    def clear_index(self):
        """Clear the index from memory and disk"""
        if self.embeddings_file.exists():
            self.embeddings_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        
        self.embeddings = None
        self.metadata = []
        self.loaded = False
        
        app_logger.info("RAG index cleared")
