import json
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from typing import Optional

from data.game_results.data_config import DataConfig
from data.game_results.transforms import shuffle_players, pad_team_players
from data.game_results.utils import create_game_result_dict, load_json, get_data_split, get_eval_game_ids


class GameResultsDataset(Dataset):
    def __init__(
        self,
        config: DataConfig,
        game_ids: list[str],
        game_results: dict,
        embeddings: dict,
    ) -> None:
        self.config = config
        self.game_ids = game_ids
        self.game_results = game_results
        self.embeddings = embeddings

    def __getitem__(self, index: int):
        game_id = self.game_ids[index]
        game = self.game_results[game_id]

        away_player_ids = shuffle_players(game["away_player_ids"], shuffle_bool=self.config.shuffle_players)
        home_player_ids = shuffle_players(game["home_player_ids"], shuffle_bool=self.config.shuffle_players)

        away_player_embeddings = [
            torch.tensor(self.embeddings[f"{player_id}_{game['season']}"]["embedding"]) for player_id in away_player_ids
        ]
        home_player_embeddings = [
            torch.tensor(self.embeddings[f"{player_id}_{game['season']}"]["embedding"]) for player_id in home_player_ids
        ]
        away_player_embeddings = pad_team_players(
            embedding_list=away_player_embeddings,
            padding_length=self.config.pad_team_players,
        )
        home_player_embeddings = pad_team_players(
            embedding_list=home_player_embeddings,
            padding_length=self.config.pad_team_players,
        )
        label = torch.tensor([1.0]) if game["home_win"] else torch.tensor([0.0])

        return torch.stack(away_player_embeddings + home_player_embeddings, dim=0), label, game_id

    def __len__(self) -> int:
        return len(self.game_ids)


class GameResultsDataModule(LightningDataModule):
    def __init__(self, config: DataConfig, stage: str = None) -> None:
        super().__init__()
        self.config = config

        self.train_dataset = None
        self.val_dataset = None
        self.setup(stage=stage)

    def setup(self, stage: Optional[str] = None):
        game_results = load_json(path=self.config.game_results_path)
        embeddings = load_json(path=self.config.embeddings_path)
        data_split = get_data_split(config=self.config, game_results=game_results, stage=stage)
        game_results = create_game_result_dict(game_results=game_results)

        self.train_dataset = GameResultsDataset(
            config=self.config,
            game_ids=data_split["train"],
            game_results=game_results,
            embeddings=embeddings,
        )
        self.val_dataset = GameResultsDataset(
            config=self.config,
            game_ids=data_split["val"],
            game_results=game_results,
            embeddings=embeddings,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size)


def load_player_season_encodings(path: str):
    with open(path, "r") as f:
        return json.load(fp=f)


def setup_data_module(config: DataConfig, stage: str = None):
    return GameResultsDataModule(config=config, stage=stage)
