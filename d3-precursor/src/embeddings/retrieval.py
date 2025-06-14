"""
Retrieval module for searching knowledge entities using embeddings.

This module provides functionality for retrieving relevant knowledge entities
based on semantic similarity using vector embeddings.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

from d3_precursor.embeddings.encoder import D3Encoder
from d3_precursor.embeddings.index import EmbeddingIndex

class D3Retriever:
    """Retrieval system for finding similar knowledge entities using embeddings.
    
    This class provides a high-level interface for retrieving relevant knowledge 
    entities based on semantic similarity through vector embeddings.
    """
    
    def __init__(
        self, 
        encoder: D3Encoder = None,
        index: EmbeddingIndex = None,
        encoder_model: str = "all-mpnet-base-v2",
        index_type: str = "flat"
    ):
        """Initialize the retriever.
        
        Args:
            encoder: D3Encoder instance for encoding text/code to vectors
            index: EmbeddingIndex instance for vector similarity search
            encoder_model: Model name to use if encoder is not provided
            index_type: Index type to use if index is not provided
        """
        self.encoder = encoder or D3Encoder(model_name=encoder_model)
        
        # Create the index if not provided
        if index is None:
            self.index = EmbeddingIndex(
                dimension=self.encoder.dimension,
                index_type=index_type
            )
        else:
            self.index = index
    
    def add_documents(
        self, 
        documents: List[str],
        metadata: List[Dict[str, Any]] = None
    ) -> List[int]:
        """Add documents to the retrieval system.
        
        Args:
            documents: List of document texts to add
            metadata: Optional metadata for each document
            
        Returns:
            List of IDs assigned to the added documents
        """
        # Encode documents
        embeddings = self.encoder.encode_batch(documents)
        
        # Add to index
        return self.index.add(embeddings, metadata)
    
    def add_items(
        self,
        items: List[Dict[str, Any]],
        text_key: str = "text",
        embed_keys: List[str] = None
    ) -> List[int]:
        """Add structured items to the retrieval system.
        
        Args:
            items: List of dictionaries containing text/code and metadata
            text_key: Key in the dictionaries that contains the text to embed
            embed_keys: Optional list of keys to embed instead of text_key
            
        Returns:
            List of IDs assigned to the added items
        """
        texts = []
        metadata = []
        
        for item in items:
            # Extract text to embed
            if embed_keys:
                # Concatenate multiple fields if specified
                text_parts = [item.get(key, "") for key in embed_keys if key in item]
                text = " ".join(text_parts)
            else:
                # Use the default text key
                text = item.get(text_key, "")
            
            texts.append(text)
            
            # Copy the item as metadata
            meta = item.copy()
            metadata.append(meta)
        
        # Encode and add to index
        embeddings = self.encoder.encode_batch(texts)
        return self.index.add(embeddings, metadata)
    
    def search(
        self, 
        query: str,
        k: int = 10,
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        """Search for the most relevant items to the query.
        
        Args:
            query: Query text to search for
            k: Number of results to return
            threshold: Optional minimum similarity threshold
            
        Returns:
            List of metadata dictionaries for the most relevant items
        """
        # Encode the query
        query_embedding = self.encoder.encode_query(query)
        
        # Search in the index
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k=k)
        
        # Get metadata for results
        metadata = self.index.get_metadata(indices)
        
        # Filter by threshold if provided
        if threshold is not None:
            results = []
            for i, meta in enumerate(metadata):
                if i < len(distances[0]) and distances[0][i] >= threshold:
                    # Add the similarity score to the metadata
                    meta = meta.copy()
                    meta["similarity"] = float(distances[0][i])
                    results.append(meta)
            return results
        else:
            # Add similarity scores
            for i, meta in enumerate(metadata):
                if i < len(distances[0]):
                    meta = meta.copy()
                    meta["similarity"] = float(distances[0][i])
            
            return metadata
    
    def batch_search(
        self,
        queries: List[str],
        k: int = 10,
        threshold: float = None
    ) -> List[List[Dict[str, Any]]]:
        """Search for multiple queries at once.
        
        Args:
            queries: List of query texts
            k: Number of results per query
            threshold: Optional minimum similarity threshold
            
        Returns:
            List of result lists, one list per query
        """
        # Encode queries
        query_embeddings = self.encoder.encode_batch(queries)
        
        # Search in the index
        distances, indices = self.index.search(query_embeddings, k=k)
        
        results = []
        for i in range(len(queries)):
            # Get metadata for results
            metadata = self.index.get_metadata(indices[i].reshape(1, -1))
            
            # Filter by threshold if provided
            if threshold is not None:
                query_results = []
                for j, meta in enumerate(metadata):
                    if j < len(distances[i]) and distances[i][j] >= threshold:
                        # Add the similarity score to the metadata
                        meta = meta.copy()
                        meta["similarity"] = float(distances[i][j])
                        query_results.append(meta)
                results.append(query_results)
            else:
                # Add similarity scores
                for j, meta in enumerate(metadata):
                    if j < len(distances[i]):
                        meta = meta.copy()
                        meta["similarity"] = float(distances[i][j])
                
                results.append(metadata)
        
        return results
    
    def save(self, directory: str):
        """Save the retriever to a directory.
        
        Args:
            directory: Directory to save the retriever to
        """
        import os
        from pathlib import Path
        
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save the index
        self.index.save(directory / "index")
    
    @classmethod
    def load(cls, directory: str, encoder: Optional[D3Encoder] = None) -> "D3Retriever":
        """Load a retriever from a directory.
        
        Args:
            directory: Directory containing a saved retriever
            encoder: Optional encoder to use instead of loading
            
        Returns:
            Loaded D3Retriever instance
        """
        from pathlib import Path
        
        directory = Path(directory)
        
        # Load the index
        index = EmbeddingIndex.load(directory / "index")
        
        # Create a new encoder if not provided
        if encoder is None:
            encoder = D3Encoder()
        
        # Return a new retriever with the loaded components
        return cls(encoder=encoder, index=index) 