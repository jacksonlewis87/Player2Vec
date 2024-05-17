from retrying import retry
from nba_api.stats.endpoints import leaguegamelog, boxscoretraditionalv3

from data.data_collection.utils import filter_dict_fields, row_to_header_dict


@retry(stop_max_attempt_number=5, wait_fixed=5000)
def get_season_game_results(season: int, filter_home_games: bool = False):
    results = leaguegamelog.LeagueGameLog(
        season=season, league_id="00", season_type_all_star="Regular Season"
    ).get_dict()["resultSets"][0]

    game_list = []
    for row in results["rowSet"]:
        game_list += [
            row_to_header_dict(
                headers=results["headers"],
                row=row,
            )
        ]

    if filter_home_games:
        game_list = [game for game in game_list if " vs. " in game["MATCHUP"]]

    return [filter_dict_fields(game, fields_to_include=["GAME_ID", "WL"]) for game in game_list]


def get_player_ids(row: list):
    return [player["personId"] for player in row]


@retry(stop_max_attempt_number=5, wait_fixed=5000)
def get_players_for_game(game_id: str):
    results = boxscoretraditionalv3.BoxScoreTraditionalV3(
        game_id=game_id,
    ).get_dict()["boxScoreTraditional"]

    return get_player_ids(results["awayTeam"]["players"]), get_player_ids(results["homeTeam"]["players"])
