import json
import os
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from typing import Optional

from constants import EndToEndStage, ROOT_DIR
from data.end_to_end.data_config import DataConfig
from data.end_to_end.transforms import (
    shuffle_players,
    pad_team_players,
    remove_total_seasons,
    normalize_dataset,
    remove_eval_seasons,
)
from data.end_to_end.utils import create_game_result_dict, load_json, get_data_split, get_eval_game_ids


class EmbeddingDataset(Dataset):
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
        with open(self.config.embedding_data_path, "r") as f:
            data = json.load(fp=f)

        data = remove_eval_seasons(dataset=data)
        data = remove_total_seasons(dataset=data)
        self.player_season_encodings = setup_player_season_encodings(data=data)
        data = {f"{row['player_id']}_{row['season']}": row for row in data}
        self.player_seasons = list(data.keys())
        self.data = normalize_dataset(dataset=data, keys_to_ignore=self.config.embedding_inference_keys_to_ignore)
        print(len(self.data))

    def __getitem__(self, index: int):
        player_season = self.player_seasons[index]
        encoding = get_one_hot_encoding(
            encoding_length=self.config.num_embeddings, encoding_value=self.player_season_encodings[player_season]
        )
        data = self.data[player_season]
        return torch.tensor(encoding), torch.tensor(data), player_season

    # def get_one_hot_encoding(self, player_season: str):
    #     encoding = get_one_hot_encoding(
    #         encoding_length=self.config.num_embeddings,
    #         encoding_value=self.player_season_encodings[player_season]
    #     )
    #     return torch.tensor(encoding)

    # def get_one_hot_encoding_with_data(self, player_season: str):
    #     encoding = get_one_hot_encoding(
    #         encoding_length=self.config.num_embeddings,
    #         encoding_value=self.player_season_encodings[player_season]
    #     )
    #     data = self.data[player_season]
    #     return torch.tensor(encoding), torch.tensor(data)

    def get_player_data(self, player_season: str):
        data = self.data[player_season]
        return torch.tensor(data)

    def __len__(self) -> int:
        return len(self.data)


# class GameResultsDataset(Dataset):
#     def __init__(
#         self,
#         config: DataConfig,
#         game_ids: list[str],
#         game_results: dict,
#     ) -> None:
#         self.config = config
#         self.game_ids = game_ids
#         self.game_results = game_results
#
#     def __getitem__(self, index: int):
#         game_id = self.game_ids[index]
#         game = self.game_results[game_id]
#
#         away_player_ids = [
#             f"{player_id}_{game['season']}" for player_id in
#             shuffle_players(game["away_player_ids"], shuffle_bool=self.config.shuffle_players)
#         ]
#         home_player_ids = [
#             f"{player_id}_{game['season']}" for player_id in
#             shuffle_players(game["home_player_ids"], shuffle_bool=self.config.shuffle_players)
#         ]
#         away_player_ids += ["" for _ in range(self.config.pad_team_players - len(away_player_ids))]
#         home_player_ids += ["" for _ in range(self.config.pad_team_players - len(home_player_ids))]
#         label = torch.tensor([1.0]) if game["home_win"] else torch.tensor([0.0])
#
#         return torch.tensor(0), label, away_player_ids, home_player_ids, game_id
#
#     def __len__(self) -> int:
#         return len(self.game_ids)


class GameResultsDataset(Dataset):
    def __init__(
        self,
        config: DataConfig,
        game_ids: list[str],
        game_results: dict,
        embeddings: dict,
        return_embeddings_data: bool = False,
    ) -> None:
        self.config = config
        self.game_ids = game_ids
        self.game_results = game_results
        self.embeddings = embeddings
        self.embedding_size = len(list(self.embeddings.values())[0]["embedding"])
        self.return_embeddings_data = return_embeddings_data
        if self.return_embeddings_data:
            self.embeddings_dataset = EmbeddingDataset(config=self.config)
            self.num_embedding_fields = len(list(self.embeddings_dataset.data.values())[0])

    def __getitem__(self, index: int):
        game_id = self.game_ids[index]
        game = self.game_results[game_id]

        # use past seasons data
        data_season = game["season"] - 1

        away_player_ids = shuffle_players(game["away_player_ids"], shuffle_bool=self.config.shuffle_players)
        home_player_ids = shuffle_players(game["home_player_ids"], shuffle_bool=self.config.shuffle_players)

        away_player_embeddings = [
            torch.tensor(self.embeddings[f"{player_id}_{data_season}"]["embedding"])
            if f"{player_id}_{data_season}" in self.embeddings
            else torch.zeros(self.embedding_size)
            for player_id in away_player_ids
        ]
        home_player_embeddings = [
            torch.tensor(self.embeddings[f"{player_id}_{data_season}"]["embedding"])
            if f"{player_id}_{data_season}" in self.embeddings
            else torch.zeros(self.embedding_size)
            for player_id in home_player_ids
        ]

        player_learnable_embeddings = []

        for i in range(self.config.pad_team_players):
            player_learnable_embeddings += [
                f"{away_player_ids[i]}_{data_season}"
                if i < len(away_player_ids) and f"{away_player_ids[i]}_{data_season}" not in self.embeddings
                else ""
            ]

        for i in range(self.config.pad_team_players):
            player_learnable_embeddings += [
                f"{home_player_ids[i]}_{data_season}"
                if i < len(home_player_ids) and f"{home_player_ids[i]}_{data_season}" not in self.embeddings
                else ""
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

        if self.return_embeddings_data:
            away_player_data = [
                torch.tensor(self.embeddings_dataset.get_player_data(f"{player_id}_{data_season}"))
                if f"{player_id}_{data_season}" in self.embeddings
                else torch.zeros(self.num_embedding_fields)
                for player_id in away_player_ids
            ]
            away_player_data += [
                torch.zeros(self.num_embedding_fields)
                for _ in range(self.config.pad_team_players - len(away_player_ids))
            ]
            home_player_data = [
                torch.tensor(self.embeddings_dataset.get_player_data(f"{player_id}_{data_season}"))
                if f"{player_id}_{data_season}" in self.embeddings
                else torch.zeros(self.num_embedding_fields)
                for player_id in home_player_ids
            ]
            home_player_data += [
                torch.zeros(self.num_embedding_fields)
                for _ in range(self.config.pad_team_players - len(home_player_ids))
            ]

            embedding_x = torch.stack(away_player_embeddings + home_player_embeddings, dim=0)
            embedding_y = torch.stack(away_player_data + home_player_data, dim=0)

        if self.return_embeddings_data:
            return (
                torch.stack(away_player_embeddings + home_player_embeddings, dim=0),
                label,
                player_learnable_embeddings,
                embedding_x,
                embedding_y,
                game_id,
            )
        else:
            return (
                torch.stack(away_player_embeddings + home_player_embeddings, dim=0),
                label,
                player_learnable_embeddings,
                game_id,
            )

    def __len__(self) -> int:
        return len(self.game_ids)


class GameResultsDataModule(LightningDataModule):
    def __init__(
        self,
        config: DataConfig,
        training_stage: str,
        stage: str = None,
    ) -> None:
        super().__init__()
        self.config = config

        self.train_dataset = None
        self.val_dataset = None
        self.setup(training_stage=training_stage, stage=stage)

    def setup(self, training_stage: str, stage: Optional[str] = None):
        if training_stage == EndToEndStage.EMBEDDING.value:
            self.setup_embedding()
        else:
            self.setup_game_results(stage=stage)

    def setup_embedding(self):
        self.train_dataset = EmbeddingDataset(config=self.config)
        self.val_dataset = []

    def setup_game_results(self, stage: Optional[str] = None):
        game_results = load_json(path=self.config.game_results_path)
        embeddings = load_json(path=self.config.embeddings_path)
        data_split = get_data_split(config=self.config, game_results=game_results, stage=stage)
        game_results = create_game_result_dict(game_results=game_results)

        self.train_dataset = GameResultsDataset(
            config=self.config,
            game_ids=data_split["train"],
            game_results=game_results,
            embeddings=embeddings,
            return_embeddings_data=True,
        )
        self.val_dataset = GameResultsDataset(
            config=self.config,
            game_ids=data_split["val"],
            game_results=game_results,
            embeddings=embeddings,
            return_embeddings_data=True,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size)


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
            json.dump(player_season_encodings, fp=f)

        return player_season_encodings


def get_one_hot_encoding(encoding_length: int, encoding_value: int):
    if encoding_value >= encoding_length:
        print(f"Error: out of range encoding {encoding_value}, {encoding_length}")
        raise Exception
    return [1.0 if i == encoding_value else 0.0 for i in range(encoding_length)]


def load_player_season_encodings(path: str):
    with open(path, "r") as f:
        return json.load(fp=f)


def setup_data_module(config: DataConfig, training_stage: str, stage: str = None):
    return GameResultsDataModule(config=config, training_stage=training_stage, stage=stage)
