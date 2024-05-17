from dataclasses import dataclass

from data.game_results.data_config import DataConfig


@dataclass
class ModelConfig:
    experiment_path: str
    learning_rate: float
    epochs: int
    checkpoint_path: str
    loss: str
    embedding_size: int
    max_num_players: int
    hidden_dim: int
    num_heads: int
    num_attention_layers: int
    dropout: float


@dataclass
class FullConfig:
    data_config: DataConfig
    model_config: ModelConfig
