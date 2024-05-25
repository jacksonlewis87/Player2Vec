from constants import ROOT_DIR
from data.data_collection.season_statlines import get_all_players, get_player_dashboard, EnhancedJSONEncoder, json

if __name__ == "__main__":
    data = []
    players = get_all_players()
    count = 0
    for player in players:
        if player.end_year >= 2000:
            print(f"{player.id}\t{count}")
            data += get_player_dashboard(player_id=player.id)
            count += 1

    with open(f"{ROOT_DIR}/data/data_dashboard_2000.json", "w") as f:
        json.dump(data, cls=EnhancedJSONEncoder, fp=f)
