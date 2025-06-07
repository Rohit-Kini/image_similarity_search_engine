from abc import ABC, abstractmethod
import torch
import numpy as np
from PIL import Image
from typing import Union, List, Tuple

class BaseImageEncoder(ABC):
    """
    Abstract base class for encoders.
    """

    def __init__(self, device: str = 'auto'):
        self.device = self._get_device(device)
        self.model = None
        self.transform = None

    def _get_device(self, device: str) -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    @abstractmethod
    def load_model(self):
        """
        Load the specific model.
        """
        pass

    @abstractmethod
    def get_transform(self):
        """
        Get image preprocessing transform.
        """
        pass

    @abstractmethod
    def encode_image(self, images: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Encode images into embedding vectors.
        
        Args:
            images Union[Image.Image, np.ndarray]: List of images to encode.
        
        Returns:
            np.ndarray: Encoded feature vectors.
        """
        pass

    def encode_batch(self, images: List[Union[Image.Image, np.ndarray]]) -> np.ndarray:
        """
        Encode a batch of images into embedding vectors.
        
        Args:
            images (List[Union[Image.Image, np.ndarray]]): List of images to encode.
        
        Returns:
            np.ndarray: Encoded feature vectors.
        """
        embeddings = []
        for image in images:
            embedding = self.encode_image(image)
            embeddings.append(embedding)
        return np.stack(embeddings)
    
    @property
    def embedding_dim(self) -> int:
        """
        Return embedding dimension.
        """
        return self._embedding_dim
    
    @property
    def model_name(self) -> str:
        """
        Return model name.
        """
        return self._model_name