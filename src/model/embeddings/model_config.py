from dataclasses import dataclass

from data.embeddings.data_config import DataConfig


@dataclass
class ModelConfig:
    experiment_path: str
    learning_rate: float
    epochs: int
    checkpoint_path: str
    loss: str
    embedding_size: int
    num_embeddings: int
    num_fields: int


@dataclass
class FullConfig:
    data_config: DataConfig
    model_config: ModelConfig
