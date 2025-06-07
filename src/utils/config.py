import yaml
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """
    Configuration for the model.
    """
    model_name: str
    model_path: str
    device: str = 'auto'

@dataclass
class DatabaseConfig:
    """
    Configuration for the vector database.
    """
    index_type: str = 'flat'
    save_path: str = 'data/embeddings/index'

@dataclass
class SearchConfig:
    """
    Configuration for the search engine.
    """
    default_k: int = 10
    batch_size: int = 32
    similarity_threshold: float = 0.5

class Config:
    """
    Central configuration manager for the application.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the configuration manager.
        
        :param config_path: Path to the YAML configuration file. If None, uses default values.
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from a YAML file or return default values.
        
        :return: Dictionary containing the configuration.
        """
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        
        return {
            'models': {
                'dinov2': {
                    'name': 'facebook/dinov2-base',
                    'device': 'auto'
                },
                'clip': {
                    'name': 'openai/clip-vit-base-patch32',
                    'device': 'auto'
                }
            },
            'database': {
                'index_type': 'flat',
                'save_path': 'data/embeddings/index'
            },
            'search': {
                'default_k': 10,
                'batch_size': 32,
                'similarity_threshold': 0.5
            }
        }
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """
        Get the configuration for a specific model.
        
        :param model_name: Name of the model.
        :return: ModelConfig instance with the model configuration.
        """
        model_config = self.config['models'][model_name]
        return ModelConfig(**model_config)
    
    def get_database_config(self) -> DatabaseConfig:
        """
        Get the configuration for the vector database.
        
        :return: DatabaseConfig instance with the database configuration.
        """
        db_config = self.config['database']
        return DatabaseConfig(**db_config)
    
    def get_search_config(self) -> SearchConfig:
        """
        Get the configuration for the search engine.
        
        :return: SearchConfig instance with the search configuration.
        """
        search_config = self.config['search']
        return SearchConfig(**search_config)