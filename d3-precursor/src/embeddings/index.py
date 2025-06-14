"""
Embedding index module for efficient similarity search.

This module implements a vector index for fast retrieval of similar embeddings.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import pickle
from pathlib import Path

class EmbeddingIndex:
    """Index for efficient similarity search over embeddings.
    
    This class provides a wrapper around vector database implementations
    to enable fast similarity search for the embedded knowledge entities.
    """
    
    def __init__(self, dimension: int = None, index_type: str = "flat"):
        """Initialize an embedding index.
        
        Args:
            dimension: Dimensionality of the embedding vectors
            index_type: Type of index to use ("flat", "hnsw", etc.)
        """
        self.dimension = dimension
        self.index_type = index_type
        self._index = None
        self._id_to_metadata = {}  # Maps IDs to original metadata
        self._next_id = 0
        
    def _init_index(self):
        """Initialize the vector index based on the chosen type."""
        try:
            import faiss
            if self.index_type == "flat":
                self._index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine if vectors are normalized)
            elif self.index_type == "hnsw":
                self._index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 neighbors
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
        except ImportError:
            raise ImportError(
                "Could not import faiss. "
                "Please install it with `pip install faiss-cpu` or `pip install faiss-gpu`."
            )
    
    @property
    def index(self):
        """Get the underlying index, initializing it if needed."""
        if self._index is None and self.dimension is not None:
            self._init_index()
        return self._index
    
    @property
    def size(self) -> int:
        """Get the number of items in the index."""
        if self.index is None:
            return 0
        return self.index.ntotal
    
    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]] = None) -> List[int]:
        """Add embeddings to the index.
        
        Args:
            embeddings: A numpy array of embeddings with shape [n, dimension]
            metadata: Optional list of metadata dictionaries associated with each embedding
            
        Returns:
            List of IDs assigned to the added embeddings
        """
        if self.dimension is None and embeddings.shape[0] > 0:
            self.dimension = embeddings.shape[1]
            self._init_index()
        
        if metadata is None:
            metadata = [{} for _ in range(embeddings.shape[0])]
        
        # Ensure we have the right number of metadata items
        if len(metadata) != embeddings.shape[0]:
            raise ValueError(f"Number of metadata items ({len(metadata)}) does not match "
                             f"number of embeddings ({embeddings.shape[0]})")
        
        # Assign IDs and store metadata
        ids = list(range(self._next_id, self._next_id + embeddings.shape[0]))
        for i, item_id in enumerate(ids):
            self._id_to_metadata[item_id] = metadata[i]
        
        self._next_id += embeddings.shape[0]
        
        # Add to the index
        self.index.add(embeddings)
        return ids
    
    def search(self, query_embeddings: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search for the k nearest embeddings to the query embeddings.
        
        Args:
            query_embeddings: Query embedding vectors with shape [n_queries, dimension]
            k: Number of nearest neighbors to retrieve
            
        Returns:
            Tuple of (distances, indices) arrays
        """
        if self.index is None or self.size == 0:
            return np.array([]), np.array([])
        
        # Ensure k doesn't exceed the index size
        k = min(k, self.size)
        
        # Perform the search
        distances, indices = self.index.search(query_embeddings, k)
        return distances, indices
    
    def get_metadata(self, indices: np.ndarray) -> List[Dict[str, Any]]:
        """Retrieve metadata for the given indices.
        
        Args:
            indices: Array of indices to retrieve metadata for
            
        Returns:
            List of metadata dictionaries
        """
        return [self._id_to_metadata.get(int(idx), {}) for idx in indices.flatten()]
    
    def save(self, directory: Union[str, Path]):
        """Save the index and metadata to a directory.
        
        Args:
            directory: Directory to save the index and metadata to
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save the index
        if self.index is not None:
            import faiss
            faiss.write_index(self.index, str(directory / "index.faiss"))
        
        # Save metadata and configuration
        with open(directory / "metadata.pkl", "wb") as f:
            pickle.dump(self._id_to_metadata, f)
        
        config = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "next_id": self._next_id
        }
        
        with open(directory / "config.json", "w") as f:
            json.dump(config, f)
    
    @classmethod
    def load(cls, directory: Union[str, Path]) -> "EmbeddingIndex":
        """Load an index from a directory.
        
        Args:
            directory: Directory containing a saved index
            
        Returns:
            Loaded EmbeddingIndex instance
        """
        directory = Path(directory)
        
        # Load configuration
        with open(directory / "config.json", "r") as f:
            config = json.load(f)
        
        index = cls(dimension=config["dimension"], index_type=config["index_type"])
        index._next_id = config["next_id"]
        
        # Load metadata
        with open(directory / "metadata.pkl", "rb") as f:
            index._id_to_metadata = pickle.load(f)
        
        # Load the index
        if (directory / "index.faiss").exists():
            import faiss
            index._index = faiss.read_index(str(directory / "index.faiss"))
        
        return index 