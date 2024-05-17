import torch
from random import shuffle


def shuffle_players(player_ids: list, shuffle_bool: bool):
    if shuffle_bool:
        shuffle(player_ids)
    return player_ids


def pad_team_players(embedding_list: list[torch.tensor], padding_length: int):
    if padding_length:
        return embedding_list + [
            torch.zeros(embedding_list[0].size()) for _ in range(padding_length - len(embedding_list))
        ]
    return embedding_list
