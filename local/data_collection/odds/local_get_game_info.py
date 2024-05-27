from retrying import retry
from nba_api.stats.endpoints import leaguegamelog

from data.data_collection.utils import row_to_header_dict


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

    game_list = [game for game in game_list if " @ " in game["MATCHUP"]]

    for game in game_list:
        print(f"{season}\t{game['GAME_DATE']}\t{game['GAME_ID']}\t{game['MATCHUP']}")


if __name__ == "__main__":
    min_season = 2017
    max_season = 2023

    while min_season <= max_season:
        get_season_game_results(season=min_season)
        min_season += 1
