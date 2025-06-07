import numpy as np
from typing import Union

class SimilarityMetrics:
    """
    Various similarity metrics for comparing embeddings.
    """

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute the cosine similarity between two vectors.
        
        :param a: First vector.
        :param b: Second vector.
        :return: Cosine similarity value.
        """
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute the Euclidean distance between two vectors.
        
        :param a: First vector.
        :param b: Second vector.
        :return: Euclidean distance value.
        """
        return np.linalg.norm(a - b)

    @staticmethod
    def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute the Manhattan distance between two vectors.
        
        :param a: First vector.
        :param b: Second vector.
        :return: Manhattan distance value.
        """
        return np.sum(np.abs(a - b))
    
    @staticmethod
    def distance_to_similarity(distance: Union[float, np.ndarray], method: str = 'exponential') -> Union[float, np.ndarray]:
        """
        Convert distance to similarity score using a specified method.
        
        :param distance: Distance value(s).
        :param method: Method to convert distance to similarity ('exponential', 'inverse', etc.).
        :return: Similarity score(s).
        """
        if method == 'exponential':
            return np.exp(-distance)
        elif method == 'inverse':
            return 1 / (1 + distance)
        elif method == 'linear':
            # Assumes normalized embeddings with max distance of ~2
            return np.maximum(0, 1 - distance / 2)
        else:
            raise ValueError(f"Unsupported method: {method}")