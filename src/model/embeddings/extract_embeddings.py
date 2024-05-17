import json
import torch

from constants import ROOT_DIR
from data.embeddings.data_module import load_player_season_encodings
from data.data_collection.season_statlines import get_all_players


def extract_embeddings(path_to_checkpoint: str, output_path: str):
    checkpoint = torch.load(path_to_checkpoint)["state_dict"]
    weights = checkpoint["model.embedding_layer.weight"]

    encodings = load_player_season_encodings(path=f"{ROOT_DIR}/data/player_season_encodings.json")
    encodings = {v: k for k, v in encodings.items()}

    players = {row.id: row.name for row in get_all_players()}
    embeddings = {}

    for i in range(weights.size(dim=1)):
        str_embedding = "\t".join([str(round(float(w), 4)) for w in list(weights[:, i])])
        player_id, season = encodings[i].split("_")
        name = players[int(player_id)]
        print(f"{encodings[i]}\t{name}\t{season}\t{str_embedding}")
        embeddings[encodings[i]] = {
            "name": name,
            "season": season,
            "embedding": [round(float(w), 4) for w in list(weights[:, i])],
        }

    with open(output_path, "w") as f:
        json.dump(embeddings, fp=f)
