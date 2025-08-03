"""
Конфигурация ML сервисов - Production Ready
"""

from dataclasses import dataclass


@dataclass
class MLServiceConfig:
    """Конфигурация ML сервиса."""

    service_url: str = "http://localhost:8001"
    models_dir: str = "./models"
    cache_dir: str = "./cache"
    max_models: int = 100
    model_timeout: int = 300
    prediction_timeout: int = 30
    enable_caching: bool = True
    cache_ttl: int = 3600
    enable_auto_scaling: bool = True
    enable_feature_engineering: bool = True
    enable_hyperparameter_optimization: bool = True
    enable_ensemble_learning: bool = True
    enable_online_learning: bool = True
    batch_size: int = 32
    learning_rate: float = 0.01
    max_iterations: int = 1000
    validation_split: float = 0.2
    test_split: float = 0.2
    random_state: int = 42
