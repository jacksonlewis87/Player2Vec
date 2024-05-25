from constants import ROOT_DIR
from data.data_collection.season_statlines import get_all_players, get_player_season, EnhancedJSONEncoder, json

if __name__ == "__main__":
    data = []
    players = get_all_players()
    for player in players:
        if player.end_year == 2023:
            data += get_player_season(player_id=player.id)

    with open(f"{ROOT_DIR}/data/data.json", "w") as f:
        json.dump(data, cls=EnhancedJSONEncoder, fp=f)
