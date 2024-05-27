import json
import os
from random import shuffle

from constants import BAD_GAME_IDS, EVAL_SEASONS
from data.game_results.data_config import DataConfig


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(fp=f)


def save_json(path: str, contents):
    with open(path, "w") as f:
        json.dump(contents, f)


def get_data_split(
    config: DataConfig, game_results: list[dict], stage: str, min_game_ids: dict, min_game_id_override: int = 0
):
    if stage == "eval":
        return {
            "train": [],
            "val": get_eval_game_ids(
                game_results=game_results, min_game_ids=min_game_ids, min_game_id_override=min_game_id_override
            ),
        }
    elif os.path.isfile(config.data_split_path):
        return load_json(path=config.data_split_path)
    else:
        return create_data_split(
            config=config,
            game_results=game_results,
            min_game_ids=min_game_ids,
            min_game_id_override=min_game_id_override,
        )


def create_data_split(config: DataConfig, game_results: list[dict], min_game_ids: dict, min_game_id_override: int = 0):
    game_ids = [
        game["game_id"]
        for game in game_results
        if game["season"] not in EVAL_SEASONS
        and game["game_id"] not in BAD_GAME_IDS
        and int(game["game_id"]) >= int(min_game_ids[game["season"]])
        and int(game["game_id"][-4:]) > min_game_id_override
    ]

    if len(list(set(game_ids))) != len(game_ids):
        print("Error: duplicate game_ids")
        raise Exception

    shuffle(game_ids)
    data_split = {
        "train": game_ids[: round(len(game_ids) * config.train_size)],
        "val": game_ids[round(len(game_ids) * config.train_size) :],
    }
    save_json(path=config.data_split_path, contents=data_split)

    return data_split


def get_eval_game_ids(game_results: list[dict], min_game_ids: dict, min_game_id_override: int = 0):
    game_ids = [
        game["game_id"]
        for game in game_results
        if game["season"] in EVAL_SEASONS
        and game["game_id"] not in BAD_GAME_IDS
        and int(game["game_id"]) >= int(min_game_ids[game["season"]])
        and int(game["game_id"][-4:]) > min_game_id_override
    ]
    return game_ids


def create_game_result_dict(game_results: list[dict]):
    game_results_dict = {}
    for game in game_results:
        game_results_dict[game["game_id"]] = {
            "season": game["season"],
            "home_win": game["home_win"],
            "away_player_ids": game["away_player_ids"],
            "home_player_ids": game["home_player_ids"],
        }

    return game_results_dict


def get_min_game_ids(embeddings: dict):
    min_game_ids = {}
    for game_id in list(set([value["game_id"] for value in embeddings.values()])):
        season = int(f"20{game_id[3:5]}")
        if season not in min_game_ids:
            min_game_ids[season] = None
        if min_game_ids[season] is None or int(min_game_ids[season][-4:]) > int(game_id[-4:]):
            min_game_ids[season] = game_id

    return min_game_ids
