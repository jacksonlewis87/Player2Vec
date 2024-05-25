import json
from constants import ROOT_DIR
from data.data_collection.games import get_season_game_results, get_players_for_game

if __name__ == "__main__":
    min_season = 2001
    max_season = 2023
    output_path = f"{ROOT_DIR}/data/game_results.json"

    existing_results = []
    try:
        with open(output_path, "r") as f:
            existing_results = json.load(fp=f)
    except Exception as e:
        print("No existing results")

    collected_games = {}
    for row in existing_results:
        if row["season"] not in collected_games:
            collected_games[row["season"]] = []
        collected_games[row["season"]] += [row["game_id"]]

    game_results = []
    while min_season <= max_season:
        print(min_season)
        games = get_season_game_results(season=min_season, filter_home_games=True)

        for game in games:
            game_id = game["GAME_ID"]
            if min_season not in collected_games or game["GAME_ID"] not in collected_games[min_season]:
                try:
                    away_player_ids, home_player_ids = get_players_for_game(game_id=game_id)
                    game_results += [
                        {
                            "season": min_season,
                            "game_id": game_id,
                            "home_win": game["WL"] == "W",
                            "away_player_ids": away_player_ids,
                            "home_player_ids": home_player_ids,
                        }
                    ]
                except Exception as e:
                    print(f"Error: {game}")
        min_season += 1

        with open(output_path, "w") as f:
            json.dump(existing_results + game_results, fp=f)
