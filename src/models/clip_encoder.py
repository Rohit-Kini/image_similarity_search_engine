import torch
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor
from PIL import Image
import numpy as np
from .base_encoder import BaseImageEncoder

class CLIPEncoder(BaseImageEncoder):
    """
    CLIPEncoder is a specific implementation of BaseImageEncoder for the CLIP model.
    It provides methods to load the model, get the image preprocessing transform,
    and encode images into embedding vectors.
    """

    def __init__(self, model_name: str = 'openai/clip-vit-base-patch32', device: str = 'auto'):
        super().__init__(device)
        self._model_name = model_name
        self._embedding_size = 512  # Default embedding size for CLIP base model
        self.load_model()

    def load_model(self):
        """
        Load the CLIP model.
        """
        self.processor = CLIPImageProcessor.from_pretrained(self._model_name)
        self.model = CLIPVisionModel.from_pretrained(self._model_name)
        self.model.to(self.device)
        self.model.eval()

    def get_transform(self):
        """
        Get the image preprocessing transform.
        
        Returns:
            CLIPImageProcessor: The image processor for CLIP.
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
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.pooler_output
        
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding.cpu().numpy().flatten()

