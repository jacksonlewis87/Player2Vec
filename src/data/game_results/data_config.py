from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    game_results_path: str
    embeddings_path: str
    data_split_path: str
    batch_size: int
    train_size: float
    shuffle_players: bool
    pad_team_players: Optional[int] = None
