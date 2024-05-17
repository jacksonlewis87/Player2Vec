import json

from constants import ROOT_DIR
from data.data_collection.season_statlines import get_all_players, get_player_profile


if __name__ == "__main__":
    data = []
    players = get_all_players()
    count = 0
    for player in players:
        if player.end_year >= 2000:
            print(f"{player.id}\t{count}")
            try:
                data += get_player_profile(player_id=player.id)
            except Exception as e:
                print("\t\t\t\tERROR")
            count += 1

    with open(f"{ROOT_DIR}/data/data_profile_2000.json", "w") as f:
        json.dump(data, fp=f)
