import faiss
import numpy as np
import pickle
import os
from typing import List, Optional, Tuple, Dict, Any
import json

class VectorDatabase:
    """
    FAISS-based vector database for storing and querying embeddings.
    """

    def __init__(self, embedding_dim: int, index_type: str = 'flat'):
        """
        Initialize the vector database with the specified embedding dimension and index type.
        
        :param embedding_dim: Dimension of the embeddings.
        :param index_type: Type of FAISS index to use (e.g., 'flat', 'IVF', etc.).
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.metadata = []
        self.is_trained = False

    def _create_index(self):
        """
        Create a FAISS index based on the specified type.
        """
        if self.index_type == 'flat':
            # Exact search using L2 distance
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == 'ivf':
            # Inverted file index for faster approximate search
            nclusters = min(100, max( 1, n_vectors // 100))
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nclusters)
        elif self.index_type == 'hnsw':
            # Hierarchical Navigable Small World graph for very fast approximate search
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]] = None):
        """
        Add embeddings and their associated metadata to the vector database.
        
        :param embeddings: Numpy array of shape (n_samples, embedding_dim) containing the embeddings.
        :param metadata: List of dictionaries containing metadata for each embedding.
        """
        if self.index is None:
            self._create_index(len(embeddings))

        # Train index if needed (for IVF)
        if self.index_type == 'ivf' and not self.is_trained:
            self.index.train(embeddings.astype)
            self.is_trained = True
        
        # Add embeddings to the index
        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for the k nearest neighbors of a query embedding.
        
        :param query_embedding: Numpy array of shape (embedding_dim,) representing the query embedding.
        :param k: Number of nearest neighbors to return.
        :return: Tuple containing indices of the nearest neighbors and their distances.
        """
        if self.index is None:
            raise RuntimeError("Index has not been created. Please add embeddings first.")
        
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query_embedding, k)
        return distances[0], indices[0]
    
    def get_metadata(self, indices: np.ndarray) -> List[Dict[str, Any]]:
        """
        Retrieve metadata for the given indices.
        
        :param indices: Numpy array of indices for which to retrieve metadata.
        :return: List of metadata dictionaries corresponding to the indices.
        """
        return [self.metadata[i] for i in indices if i < len(self.metadata)]
    
    def save(self, filepath: str):
        """
        Save the vector database to a file.
        
        :param filepath: Path to the file where the database will be saved.
        """
        if self.index is None:
            raise RuntimeError("Index has not been created. Please add embeddings first.")
        
        faiss.write_index(self.index, filepath + '.index')

        # Save metadata and config
        config = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'is_trained': self.is_trained,
            'metadata': self.metadata
        }

        with open(filepath + '.pkl', 'wb') as f:
            pickle.dump(config, f)

    def load(self, filepath: str):
        """
        Load the vector database from a file.
        
        :param filepath: Path to the file from which the database will be loaded.
        """
        self.index = faiss.read_index(filepath + '.index')

        # Load metadata and config
        with open(filepath + '.pkl', 'rb') as f:
            config = pickle.load(f)
        
        self.embedding_dim = config['embedding_dim']
        self.index_type = config['index_type']
        self.is_trained = config['is_trained']
        self.metadata = config['metadata']
        
        # TODO: Recreate the index if necessary
        if not self.is_trained and self.index_type == 'ivf':
            raise RuntimeError("Index is not trained. Please add embeddings before searching.")