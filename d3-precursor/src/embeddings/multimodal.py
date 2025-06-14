"""
Multimodal embeddings module for handling different types of data.

This module provides functionality for encoding different types of data 
(text, code, images, etc.) into vector embeddings and combining them.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from pathlib import Path
import os
import logging

from d3_precursor.embeddings.encoder import D3Encoder

logger = logging.getLogger(__name__)

class MultimodalEncoder:
    """Encoder for handling multiple modalities of data.
    
    This class provides methods for encoding different types of data
    into vector embeddings and combining them in meaningful ways.
    """
    
    def __init__(
        self,
        text_model: str = "all-mpnet-base-v2",
        code_model: str = "microsoft/codebert-base",
        image_model: Optional[str] = None,
        embedding_fusion: str = "concatenate"
    ):
        """Initialize the multimodal encoder.
        
        Args:
            text_model: Model name for text encoding
            code_model: Model name for code encoding
            image_model: Optional model name for image encoding
            embedding_fusion: Strategy for combining embeddings
                ("concatenate", "average", "weighted")
        """
        self.embedding_fusion = embedding_fusion
        
        # Initialize text encoder
        self.text_encoder = D3Encoder(model_name=text_model)
        
        # Initialize code encoder if different from text
        if code_model != text_model:
            self.code_encoder = D3Encoder(model_name=code_model)
        else:
            self.code_encoder = self.text_encoder
        
        # Initialize image encoder if provided
        self.image_encoder = None
        if image_model:
            try:
                from transformers import CLIPProcessor, CLIPModel
                self.image_model = CLIPModel.from_pretrained(image_model)
                self.image_processor = CLIPProcessor.from_pretrained(image_model)
                self.has_image_support = True
            except ImportError:
                logger.warning(
                    "Could not import transformers for image encoding. "
                    "Install with 'pip install transformers torch Pillow'"
                )
                self.has_image_support = False
            except Exception as e:
                logger.error(f"Error loading image model: {e}")
                self.has_image_support = False
        else:
            self.has_image_support = False
        
        # Calculate combined dimension
        self._calculate_dimensions()
    
    def _calculate_dimensions(self):
        """Calculate dimensions for the combined embeddings."""
        self.text_dimension = self.text_encoder.dimension
        self.code_dimension = self.code_encoder.dimension
        
        if self.has_image_support:
            # CLIP models typically have 512 dimensions
            self.image_dimension = 512
        else:
            self.image_dimension = 0
        
        if self.embedding_fusion == "concatenate":
            self.dimension = self.text_dimension + self.code_dimension
            if self.has_image_support:
                self.dimension += self.image_dimension
        else:
            # For average or weighted fusion, use the max dimension
            self.dimension = max(self.text_dimension, self.code_dimension)
            if self.has_image_support:
                self.dimension = max(self.dimension, self.image_dimension)
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Encode text data.
        
        Args:
            text: Text string or list of strings
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(text, str):
            return self.text_encoder.encode(text)
        else:
            return self.text_encoder.encode_batch(text)
    
    def encode_code(self, code: Union[str, List[str]]) -> np.ndarray:
        """Encode code data.
        
        Args:
            code: Code string or list of strings
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(code, str):
            return self.code_encoder.encode(code)
        else:
            return self.code_encoder.encode_batch(code)
    
    def encode_image(self, image_path: Union[str, Path, List[Union[str, Path]]]) -> np.ndarray:
        """Encode image data.
        
        Args:
            image_path: Path to image file or list of paths
            
        Returns:
            Numpy array of embeddings
        """
        if not self.has_image_support:
            raise ValueError("Image encoding is not available. Install required dependencies.")
        
        import torch
        from PIL import Image
        
        # Handle single image
        if isinstance(image_path, (str, Path)):
            try:
                image = Image.open(image_path).convert('RGB')
                inputs = self.image_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.image_model.get_image_features(**inputs)
                return outputs.cpu().numpy()[0]
            except Exception as e:
                logger.error(f"Error encoding image {image_path}: {e}")
                # Return zeros as fallback
                return np.zeros(self.image_dimension)
        
        # Handle multiple images
        embeddings = []
        for path in image_path:
            try:
                image = Image.open(path).convert('RGB')
                inputs = self.image_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.image_model.get_image_features(**inputs)
                embeddings.append(outputs.cpu().numpy()[0])
            except Exception as e:
                logger.error(f"Error encoding image {path}: {e}")
                # Use zeros as fallback
                embeddings.append(np.zeros(self.image_dimension))
        
        return np.array(embeddings)
    
    def combine_embeddings(
        self, 
        embeddings: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """Combine multiple embeddings into one.
        
        Args:
            embeddings: List of embedding arrays
            weights: Optional weights for weighted average
            
        Returns:
            Combined embedding
        """
        if not embeddings:
            raise ValueError("No embeddings provided to combine")
        
        if len(embeddings) == 1:
            return embeddings[0]
        
        if self.embedding_fusion == "concatenate":
            return np.concatenate(embeddings)
        
        elif self.embedding_fusion == "average":
            # Normalize each embedding first
            normalized = []
            for emb in embeddings:
                norm = np.linalg.norm(emb)
                if norm > 0:
                    normalized.append(emb / norm)
                else:
                    normalized.append(emb)
            
            # Take the mean
            return np.mean(normalized, axis=0)
        
        elif self.embedding_fusion == "weighted":
            if weights is None:
                # Equal weights if not provided
                weights = [1.0 / len(embeddings)] * len(embeddings)
            elif len(weights) != len(embeddings):
                raise ValueError("Number of weights must match number of embeddings")
            
            # Normalize weights
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
            else:
                weights = [1.0 / len(embeddings)] * len(embeddings)
            
            # Weighted sum
            result = np.zeros_like(embeddings[0])
            for i, emb in enumerate(embeddings):
                # Normalize the embedding
                norm = np.linalg.norm(emb)
                if norm > 0:
                    result += weights[i] * (emb / norm)
                else:
                    result += weights[i] * emb
            
            return result
        
        else:
            raise ValueError(f"Unknown fusion method: {self.embedding_fusion}")
    
    def encode_multimodal(
        self,
        text: Optional[str] = None,
        code: Optional[str] = None,
        image_path: Optional[Union[str, Path]] = None,
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """Encode multiple modalities into a single embedding.
        
        Args:
            text: Optional text content
            code: Optional code content
            image_path: Optional path to image
            weights: Optional weights for combining embeddings
            
        Returns:
            Combined embedding vector
        """
        embeddings = []
        
        if text:
            text_embedding = self.encode_text(text)
            embeddings.append(text_embedding)
        
        if code:
            code_embedding = self.encode_code(code)
            embeddings.append(code_embedding)
        
        if image_path and self.has_image_support:
            image_embedding = self.encode_image(image_path)
            embeddings.append(image_embedding)
        
        if not embeddings:
            raise ValueError("At least one modality must be provided")
        
        return self.combine_embeddings(embeddings, weights)
    
    def encode_batch_multimodal(
        self,
        items: List[Dict[str, Any]],
        text_key: str = "text",
        code_key: str = "code",
        image_key: str = "image_path",
        weights_key: str = "modality_weights"
    ) -> np.ndarray:
        """Encode a batch of multimodal items.
        
        Args:
            items: List of dictionaries with different modality content
            text_key: Dictionary key for text content
            code_key: Dictionary key for code content
            image_key: Dictionary key for image paths
            weights_key: Dictionary key for modality weights
            
        Returns:
            Array of combined embeddings
        """
        all_embeddings = []
        
        for item in items:
            embeddings = []
            
            # Extract text if available
            if text_key in item and item[text_key]:
                text_embedding = self.encode_text(item[text_key])
                embeddings.append(text_embedding)
            
            # Extract code if available
            if code_key in item and item[code_key]:
                code_embedding = self.encode_code(item[code_key])
                embeddings.append(code_embedding)
            
            # Extract image if available
            if image_key in item and item[image_key] and self.has_image_support:
                image_embedding = self.encode_image(item[image_key])
                embeddings.append(image_embedding)
            
            # Get weights if available
            weights = item.get(weights_key)
            
            if not embeddings:
                logger.warning(f"No valid content found for item: {item}")
                # Use zeros as fallback
                all_embeddings.append(np.zeros(self.dimension))
            else:
                # Combine the embeddings
                combined = self.combine_embeddings(embeddings, weights)
                all_embeddings.append(combined)
        
        return np.array(all_embeddings)
    
    def save(self, directory: str):
        """Save the encoder configuration.
        
        Args:
            directory: Directory to save the configuration to
        """
        import json
        from pathlib import Path
        
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        config = {
            "text_model": self.text_encoder.model_name,
            "code_model": self.code_encoder.model_name,
            "image_model": getattr(self, "image_model_name", None),
            "embedding_fusion": self.embedding_fusion,
            "has_image_support": self.has_image_support,
            "dimension": self.dimension
        }
        
        with open(directory / "multimodal_config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, directory: str) -> "MultimodalEncoder":
        """Load a multimodal encoder from a directory.
        
        Args:
            directory: Directory containing the saved configuration
            
        Returns:
            MultimodalEncoder instance
        """
        import json
        from pathlib import Path
        
        directory = Path(directory)
        
        try:
            with open(directory / "multimodal_config.json", "r") as f:
                config = json.load(f)
            
            return cls(
                text_model=config["text_model"],
                code_model=config["code_model"],
                image_model=config["image_model"],
                embedding_fusion=config["embedding_fusion"]
            )
        except Exception as e:
            logger.error(f"Error loading multimodal encoder: {e}")
            # Return default encoder
            return cls() 