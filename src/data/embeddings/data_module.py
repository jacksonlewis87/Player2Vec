import json
import os
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from typing import Optional

from constants import ROOT_DIR
from data.data_collection.season_statlines import EnhancedJSONEncoder
from data.embeddings.transforms import remove_total_seasons, normalize_dataset, remove_eval_seasons
from data.embeddings.data_config import DataConfig


class Player2VecDataset(Dataset):
    def __init__(
        self,
        config: DataConfig,
    ) -> None:
        self.config = config
        self.player_seasons = []
        self.player_season_encodings = {}
        self.data = {}
        self.setup()

    def setup(self):
        with open(self.config.data_path, "r") as f:
            data = json.load(fp=f)

        data = remove_eval_seasons(dataset=data)
        data = remove_total_seasons(dataset=data)
        self.player_season_encodings = setup_player_season_encodings(data=data)
        data = {f"{row['player_id']}_{row['season']}": row for row in data}
        self.player_seasons = list(data.keys())
        self.data = normalize_dataset(dataset=data, keys_to_ignore=self.config.keys_to_ignore)
        print(len(self.data))

    def __getitem__(self, index: int):
        encoding = get_one_hot_encoding(
            encoding_length=len(self.player_season_encodings),
            encoding_value=self.player_season_encodings[self.player_seasons[index]],
        )
        data = self.data[self.player_seasons[index]]
        return torch.tensor(encoding), torch.tensor(data)

    def __len__(self) -> int:
        return len(self.data)


class Player2VecDataModule(LightningDataModule):
    def __init__(
        self,
        config: DataConfig,
    ) -> None:
        super().__init__()
        self.config = config

        self.train_dataset = None
        self.setup()

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = Player2VecDataset(config=self.config)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(None, batch_size=self.config.batch_size)


def load_player_season_encodings(path: str):
    with open(path, "r") as f:
        return json.load(fp=f)


def setup_player_season_encodings(data: list[dict]):
    path = f"{ROOT_DIR}/data/player_season_encodings.json"
    if os.path.isfile(path):
        return load_player_season_encodings(path=path)
    else:
        player_seasons = [f"{row['player_id']}_{row['season']}" for row in data]
        if len(player_seasons) != len(list(set(player_seasons))):
            print(len(player_seasons))
            print(len(list(set(player_seasons))))
            raise Exception
        player_seasons.sort()
        player_season_encodings = {}
        for i, ps in enumerate(player_seasons):
            player_season_encodings[ps] = i

        with open(f"{ROOT_DIR}/data/player_season_encodings.json", "w") as f:
            json.dump(player_season_encodings, cls=EnhancedJSONEncoder, fp=f)

        return player_season_encodings


def get_one_hot_encoding(encoding_length: int, encoding_value: int):
    if encoding_value >= encoding_length:
        print("Error: out of range encoding")
        raise Exception
    return [1.0 if i == encoding_value else 0.0 for i in range(encoding_length)]


def setup_data_module(config: DataConfig):
    return Player2VecDataModule(config=config)
