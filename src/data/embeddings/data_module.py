import json
import os
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from typing import Optional

from data.embeddings.transforms import (
    remove_non_total_seasons,
    normalize_dataset,
    remove_eval_seasons,
    remove_eval_seasons_game_id,
    convert_to_list,
)
from model.embeddings.model_config import FullConfig


class Player2VecDataset(Dataset):
    def __init__(
        self,
        config: FullConfig,
    ) -> None:
        self.config = config
        self.player_seasons = []
        self.player_season_encodings = {}
        self.data = {}
        self.setup()

    def setup(self):
        with open(self.config.data_config.data_path, "r") as f:
            data = json.load(fp=f)

        if isinstance(data, dict):
            data = remove_eval_seasons_game_id(dataset=data)
            data = convert_to_list(dataset=data, key_field_names=["game_id", "player_id"])
            row_to_id = lambda row: f"{row['player_id']}_{row['game_id']}"
        else:
            data = remove_eval_seasons(dataset=data)
            data = remove_non_total_seasons(dataset=data)
            row_to_id = lambda row: f"{row['player_id']}_{row['season']}"

        self.player_season_encodings = setup_player_season_encodings(
            data=data,
            experiment_path=self.config.model_config.experiment_path,
            row_to_id=row_to_id,
        )
        data = {row_to_id(row): row for row in data}
        self.player_seasons = list(data.keys())
        self.data = normalize_dataset(dataset=data, keys_to_ignore=self.config.data_config.keys_to_ignore)
        print(len(self.data))

    def __getitem__(self, index: int):
        encoding = self.player_season_encodings[self.player_seasons[index]]
        data = self.data[self.player_seasons[index]]
        return torch.tensor(encoding), torch.tensor(data)

    def __len__(self) -> int:
        return len(self.data)


class Player2VecDataModule(LightningDataModule):
    def __init__(
        self,
        config: FullConfig,
    ) -> None:
        super().__init__()
        self.config = config

        self.train_dataset = None
        self.setup()

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = Player2VecDataset(config=self.config)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.config.data_config.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(None, batch_size=self.config.data_config.batch_size, shuffle=False)


def load_player_season_encodings(path: str):
    with open(path, "r") as f:
        return json.load(fp=f)


def setup_player_season_encodings(data: list[dict], experiment_path: str, row_to_id):
    path = f"{experiment_path}/player_season_encodings.json"
    if os.path.isfile(path):
        return load_player_season_encodings(path=path)
    else:
        player_seasons = [row_to_id(row) for row in data]
        if len(player_seasons) != len(list(set(player_seasons))):
            print(len(player_seasons))
            print(len(list(set(player_seasons))))
            raise Exception
        player_seasons.sort()
        player_season_encodings = {}
        for i, ps in enumerate(player_seasons):
            player_season_encodings[ps] = i

        os.makedirs(f"{experiment_path}", exist_ok=True)
        with open(f"{experiment_path}/player_season_encodings.json", "w") as f:
            json.dump(player_season_encodings, fp=f)

        return player_season_encodings


def get_one_hot_encoding(encoding_length: int, encoding_value: int):
    if encoding_value >= encoding_length:
        print("Error: out of range encoding")
        raise Exception
    return [1.0 if i == encoding_value else 0.0 for i in range(encoding_length)]


def setup_data_module(config: FullConfig):
    return Player2VecDataModule(config=config)
