from datetime import datetime, timedelta
from retrying import retry
from nba_api.stats.endpoints import leaguegamelog, playergamelog

from data.data_collection.utils import row_to_header_dict, sum_rows


@retry(stop_max_attempt_number=5, wait_fixed=5000)
def get_season_game_dates(season: int):
    results = leaguegamelog.LeagueGameLog(
        season=season, league_id="00", season_type_all_star="Regular Season"
    ).get_dict()["resultSets"][0]

    game_dates = {}

    for row in results["rowSet"]:
        game_dates[row[4]] = row[5]

    return game_dates


# @retry(stop_max_attempt_number=5, wait_fixed=5000)
def get_player_stats(player_id: int, date: str, date_diff: int):
    # Cant use current date of game being played
    day_offset = 1
    query_date = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=day_offset)

    # Construct start and end dates for the query (last 30 days)
    end_date = query_date.strftime("%m/%d/%Y")
    start_date = (query_date - timedelta(days=date_diff + day_offset)).strftime("%m/%d/%Y")

    # Get player game log
    player_game_log = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=f"{int(date[:4]) - 1}-{date[2:4]}",
        season_type_all_star="Regular Season",
        date_from_nullable=start_date,
        date_to_nullable=end_date,
    ).get_dict()["resultSets"][0]

    trim_start = 6
    trim_end = 1

    games = player_game_log["rowSet"]
    headers = ["GP"] + player_game_log["headers"][trim_start : trim_end * -1]

    if len(games) > 0:
        stat_totals = [len(games)] + sum_rows(list_of_rows=games, trim_start=trim_start, trim_end=trim_end)
    else:
        stat_totals = [0.0 for _ in range(len(headers))]

    # Extract relevant stats
    stats = row_to_header_dict(headers=headers, row=stat_totals)

    # Update shooting percentages
    for field in headers:
        if field[-4:] == "_PCT":
            stat_name = field[:-4]
            if stats[f"{stat_name}A"] > 0:
                stats[field] = round(stats[f"{stat_name}M"] / stats[f"{stat_name}A"], 3)
            else:
                stats[field] = 0.0

    return stats
