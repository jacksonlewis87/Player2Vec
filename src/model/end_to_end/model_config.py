from dataclasses import dataclass

from data.end_to_end.data_config import DataConfig


@dataclass
class ModelConfig:
    experiment_path: str
    learning_rate: float
    epochs: int
    checkpoint_path: str
    state_dict_path: str
    embedding_size: int
    num_fields: int
    hidden_dim: int
    num_heads: int
    num_attention_layers: int
    dropout: float
    training_stage: str
    embedding_loss_weight: float


@dataclass
class FullConfig:
    data_config: DataConfig
    model_config: ModelConfig
