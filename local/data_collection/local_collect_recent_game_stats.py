from constants import ROOT_DIR

from data.data_collection.recent_game_stats import get_season_game_dates, get_player_stats
from utils import load_json, write_json


if __name__ == "__main__":
    save_every_n_games = 20
    date_diff = 30
    output_path = f"{ROOT_DIR}/data/recent_game_stats_{date_diff}.json"
    game_results_path = f"{ROOT_DIR}/data/game_results.json"

    existing_results = {}
    try:
        existing_results = load_json(path=output_path)
    except Exception as e:
        print("No existing results")

    game_results = load_json(path=game_results_path)
    game_date_mappings = {}

    save_count = save_every_n_games

    for game in game_results:
        if game["game_id"] in existing_results:
            continue

        try:
            if game["game_id"] not in game_date_mappings:
                game_date_mappings = {**game_date_mappings, **get_season_game_dates(season=game["season"])}

            game_date = game_date_mappings[game["game_id"]]

            # Don't collect games early in the season
            if int(game_date[:4]) == game["season"]:
                continue

            print(game["game_id"])

            game_players = {}

            for player_id in game["away_player_ids"] + game["home_player_ids"]:
                player_stats = get_player_stats(player_id=player_id, date=game_date, date_diff=date_diff)
                game_players[player_id] = player_stats

            existing_results[game["game_id"]] = game_players

            if save_count <= 0:
                print("\tsaving")
                write_json(path=output_path, json_object=existing_results)
                save_count = save_every_n_games
            else:
                save_count -= 1
        except Exception as e:
            print(e)
            print("\tError")

    write_json(path=output_path, json_object=existing_results)
