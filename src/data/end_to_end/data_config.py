from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    embeddings_path: str
    embedding_data_path: str
    game_results_path: str
    data_split_path: str
    batch_size: int
    train_size: float
    shuffle_players: bool
    embedding_inference_keys_to_ignore: list[str]
    num_embeddings: int
    pad_team_players: Optional[int] = None
