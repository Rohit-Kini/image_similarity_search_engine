import torch
import torch.nn.functional as F
from transformers import Dinov2Model, Dinov2ImageProcessor
from PIL import Image
from numpy as np
from .base_encoder import BaseImageEncoder

class DinoV2Encoder(BaseImageEncoder):
    """
    DinoV2Encoder is a specific implementation of BaseImageEncoder for the DinoV2 model.
    It provides methods to load the model, get the image preprocessing transform,
    and encode images into embedding vectors.
    """

    def __init__(self, model_name: str = 'facebook/dinov2-base', device: str = 'auto'):
        super().__init__(device)
        self._model_name = model_name
        self._embedding_size = 768 # Base model dimension
        self.load_model()

    def load_model(self):
        """
        Load the DinoV2 model.
        """
        self.processor = Dinov2ImageProcessor.from_pretrained(self._model_name)
        self.model = Dinov2Model.from_pretrained(self._model_name)
        self.model.to(self.device)
        self.model.eval()

        # Update embedding size based on the model configuration
        if 'large' in self._model_name:
            self._embedding_size = 1024
        elif 'giant' in self._model_name:
            self._embedding_size = 1536

    def get_transform(self):
        """
        Get the image preprocessing transform.
        
        Returns:
            Dinov2ImageProcessor: The image processor for DinoV2.
        """
        return self.processor
    
    def encode_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Encode a single image into an embedding vector.
        
        Args:
            image (Union[Image.Image, np.ndarray]): Image to encode.
        
        Returns:
            np.ndarray: Encoded feature vector.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Preprocess the image using the processor
        inputs = self.processor(image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings from the model
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0] # Shape: [1, embedding_size]
        
        #Normalize the embedding
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy().flatten()
    
