import numpy as np
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import time

from src.models.base_encoder import BaseImageEncoder
from src.search.vector_db import VectorDatabase
from src.search.similarity_metrics import SimilarityMetrics

class ImageSearchEngine:
    """
    Main class for managing image search operations.
    """

    def __init__(self, encoder: BaseImageEncoder, db_config: Dict[str, Any]=None):
        """
        Initialize the image search engine with an encoder and database configuration.
        
        :param encoder: An instance of BaseImageEncoder for encoding images.
        :param db_config: Configuration dictionary for the vector database.
        """
        self.encoder = encoder
        self.db_config = db_config or {'index_type': 'flat'}
        self.database = VectorDatabase(embedding_dim=encoder.embedding_dim,
                                       index_type=self.db_config['index_type'])
        self.similarity_metrics = SimilarityMetrics()

    def build_index(self, image_paths: List[str], batch_size: int = 32, save_path: Optional[str] = None):
        """
        Build the vector database index from a list of image paths.
        
        :param image_paths: List of file paths to images.
        :param batch_size: Number of images to process in each batch.
        :param save_path: Optional path to save the index after building.
        """
        embeddings = []
        metadata = []
        failed_images = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            batch_metadata = []

            # Load batch images
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    batch_images.append(image)
                    batch_metadata.append({
                        'path': path,
                        'filename': os.path.basename(path),
                        'index': len(metadata) + len(batch_metadata)
                    })
                except Exception as e:
                    print(f"Failed to load image {path}: {e}")
                    failed_images.append(path)
                    continue

                if batch_images:
                    batch_embeddings = self.encoder.encode_batch(batch_images)
                    embeddings.extend(batch_embeddings)
                    metadata.extend(batch_metadata)

                print(f"Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)}")

        # Add to database
        if embeddings:
            embeddings_array = np.array(embeddings)
            self.database.add_embeddings(embeddings_array, metadata)
        
            if save_path:
                self.database.save(save_path)
                print(f"Index saved to {save_path}")
        
        print(f"Index building complete. {len(embeddings)} images indexed, {len(failed_images)} failed.")
        return len(embeddings), failed_images
    
    def search_by_image(self, query_image: Image.Image, k: int = 10, return_scores: bool = True) -> List[Dict[str, Any]]:
        """
        Search for similar images based on a query image.
        
        :param query_image: Image to search for similar images.
        :param k: Number of nearest neighbors to return.
        :param return_scores: Whether to return similarity scores.
        :return: List of dictionaries containing metadata and optionally scores.
        """
        if isinstance(query_image, str):
            query_image = Image.open(query_image).convert('RGB')

        # Encode the query image
        start_time = time.time()
        query_embedding = self.encoder.encode(query_image)
        encode_time = time.time() - start_time

        # Search the database
        start_time = time.time()
        distances, indices = self.database.search(query_embedding, k)
        search_time = time.time() - start_time

        # Get metadata and format results
        results_metadata = self.database.get_metadata(indices)
        results = []

        for i, (distance, metadata) in enumerate(zip(distances, results_metadata)):
            result = {
                'rank': i + 1,
                'distance': distance,
                'similarity': self.similarity_metrics.distance_to_similarity(distance),
                'metadata': metadata
            }

            if return_scores:
                result.update({
                    'encode_time': encode_time,
                    'search_time': search_time,
                })
            results.append(result)

        return results

    def load_index(self, filepath: str):
            """
            Load pre-built index from a file.

            :param filepath: Path to the file containing the index.
            """
            self.database.load(filepath)
            print(f"Loaded index with {len(self.database.metadata)} images from {filepath}")

    def get_Stats(self) -> Dict[str, Any]:
        """
        Get statistics about the search engine
        
        :return: Dictionary containing search engine statistics.
        """
        return {
            'total_images': len(self.database.metadata),
            'embedding_dim': self.encoder.embedding_dim,
            'model_name': self.encoder.model_name,
            'index_type': self.database.index_type,
            'device': self.encoder.device
        }