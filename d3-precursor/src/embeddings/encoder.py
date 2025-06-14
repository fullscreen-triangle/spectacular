"""
Encoder module that provides vector embeddings for D3 knowledge entities.

This module implements the base encoder class that converts text/code into vector embeddings.
"""

import numpy as np
from typing import List, Union, Dict, Any, Optional
import torch

class D3Encoder:
    """Base embedding encoder for D3 knowledge entities.
    
    This class wraps various embedding models to provide a consistent interface
    for converting text and code snippets into vector representations.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 device: Optional[str] = None,
                 max_length: int = 512,
                 normalize_embeddings: bool = True):
        """Initialize the encoder with a specific embedding model.
        
        Args:
            model_name: The name of the model to use for embeddings
            device: The device to run the model on (cpu, cuda, etc.)
            max_length: Maximum sequence length for the model
            normalize_embeddings: Whether to L2-normalize the output embeddings
        """
        self.model_name = model_name
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Load model (lazy loading)
        self._model = None
        
    def _load_model(self):
        """Lazy load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except ImportError:
            raise ImportError(
                "Could not import SentenceTransformer. "
                "Please install it with `pip install sentence-transformers`."
            )
    
    @property
    def model(self):
        """Get the underlying model, loading it if needed."""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embeddings produced by this encoder."""
        return self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Encode the given texts into vector embeddings.
        
        Args:
            texts: A string or list of strings to encode
            batch_size: The batch size to use when encoding
            
        Returns:
            A numpy array of embeddings, with shape [n_texts, embedding_dim]
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings
        )
    
    def encode_queries(self, queries: Union[str, List[str]]) -> np.ndarray:
        """Encode queries (which might be structured differently than documents).
        
        Args:
            queries: A string or list of strings representing queries
            
        Returns:
            A numpy array of embeddings
        """
        # By default, queries are encoded the same way as documents
        return self.encode(queries)
    
    def encode_api(self, api_docs: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Encode D3 API documentation into vector embeddings.
        
        Args:
            api_docs: List of API documentation items
            
        Returns:
            Dictionary mapping API names to embeddings
        """
        api_texts = []
        api_names = []
        
        for doc in api_docs:
            if 'name' in doc and ('description' in doc or 'docstring' in doc):
                formatted_doc = self._format_api_for_embedding(doc)
                api_texts.append(formatted_doc)
                api_names.append(doc['name'])
        
        # Encode the formatted API texts
        embeddings = self.encode(api_texts)
        
        # Map API names to their embeddings
        result = {}
        for i, name in enumerate(api_names):
            result[name] = embeddings[i]
            
        return result
    
    def _format_api_for_embedding(self, api_doc: Dict[str, Any]) -> str:
        """Format API documentation for embedding.
        
        This method creates a standardized text representation of API documentation
        that is optimized for semantic similarity search.
        
        Args:
            api_doc: API documentation dictionary
            
        Returns:
            Formatted text representation for embedding
        """
        parts = []
        
        # Add API name and namespace
        full_name = api_doc.get('full_name', api_doc.get('name', ''))
        namespace = api_doc.get('namespace', '')
        
        if full_name:
            parts.append(f"API: {full_name}")
        
        if namespace and namespace not in full_name:
            parts.append(f"Namespace: {namespace}")
            
        # Add API type
        api_type = api_doc.get('type', '')
        if api_type:
            parts.append(f"Type: {api_type}")
            
        # Add signature if available
        signature = api_doc.get('signature', '')
        if signature:
            parts.append(f"Signature: {signature}")
            
        # Add description or docstring
        description = api_doc.get('description', api_doc.get('docstring', ''))
        if description:
            parts.append(f"Description: {description}")
            
        # Add parameters
        params = api_doc.get('parameters', api_doc.get('param_list', []))
        if params:
            param_texts = []
            for param in params:
                param_name = param.get('name', '')
                param_desc = param.get('description', '')
                if param_name:
                    if param_desc:
                        param_texts.append(f"{param_name}: {param_desc}")
                    else:
                        param_texts.append(param_name)
                        
            if param_texts:
                parts.append(f"Parameters: {', '.join(param_texts)}")
                
        # Add return value
        returns = api_doc.get('returns', {})
        if returns:
            return_type = returns.get('type', '')
            return_desc = returns.get('description', '')
            
            if return_type and return_desc:
                parts.append(f"Returns: {return_type} - {return_desc}")
            elif return_desc:
                parts.append(f"Returns: {return_desc}")
                
        # Add examples
        examples = api_doc.get('examples', [])
        if examples and len(examples) > 0:
            # Include just the first example to keep the embedding focused
            example = examples[0].get('code', '')
            if example:
                # Truncate long examples
                if len(example) > 500:
                    example = example[:500] + "..."
                parts.append(f"Example: {example}")
                
        # Add usage patterns
        patterns = api_doc.get('patterns', [])
        if patterns and len(patterns) > 0:
            pattern_texts = [p.get('pattern', '') for p in patterns if 'pattern' in p]
            if pattern_texts:
                parts.append(f"Common patterns: {', '.join(pattern_texts[:3])}")
        
        # Join all parts with newlines to create the final text
        return "\n\n".join(parts)
        
    def encode_patterns(self, patterns: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Encode D3 usage patterns into vector embeddings.
        
        Args:
            patterns: List of pattern dictionaries
            
        Returns:
            Dictionary mapping pattern descriptions to embeddings
        """
        pattern_texts = []
        pattern_ids = []
        
        for i, pattern in enumerate(patterns):
            if 'pattern' in pattern and 'code' in pattern:
                text = f"Pattern: {pattern['pattern']}\nCode: {pattern['code']}"
                pattern_texts.append(text)
                pattern_ids.append(i)
        
        # Encode the pattern texts
        embeddings = self.encode(pattern_texts)
        
        # Map pattern IDs to their embeddings
        result = {}
        for i, idx in enumerate(pattern_ids):
            result[idx] = embeddings[i]
            
        return result 