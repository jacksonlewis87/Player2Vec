from dataclasses import dataclass
from retrying import retry
from nba_api.stats.endpoints import commonallplayers, playerprofilev2


@dataclass
class Player:
    id: int
    name: str
    roster_status: int
    start_year: int
    end_year: int


def get_all_players():
    player_list = []
    players = commonallplayers.CommonAllPlayers()
    for row in players.get_dict()["resultSets"][0]["rowSet"]:
        player_list += [
            Player(
                id=row[0],
                name=row[2],
                roster_status=row[3],
                start_year=int(row[4]),
                end_year=int(row[5]),
            )
        ]
    return player_list


@retry(stop_max_attempt_number=5, wait_fixed=5000)
def get_player_profile(player_id: int):
    season_list = {}
    career = playerprofilev2.PlayerProfileV2(player_id=player_id).get_dict()
    for result_set in career["resultSets"]:
        if result_set["name"] == "SeasonTotalsRegularSeason":
            for row in result_set["rowSet"]:
                if f"{player_id}_{int(row[1].split('-')[0])}" not in season_list:
                    season_list[f"{player_id}_{int(row[1].split('-')[0])}"] = []
                season_list[f"{player_id}_{int(row[1].split('-')[0])}"] += [
                    {
                        "player_id": player_id,
                        "season": int(row[1].split("-")[0]),
                        "team_id": row[3],
                        "player_age": row[5],
                        "gp": row[6],
                        "gs": row[7],
                        "min": row[8],
                        "fgm": row[9],
                        "fga": row[10],
                        "fg_pct": row[11],
                        "fg3m": row[12],
                        "fg3a": row[13],
                        "fg3_pct": row[14],
                        "ftm": row[15],
                        "fta": row[16],
                        "ft_pct": row[17],
                        "oreb": row[18],
                        "dreb": row[19],
                        "reb": row[20],
                        "ast": row[21],
                        "stl": row[22],
                        "blk": row[23],
                        "tov": row[24],
                        "pf": row[25],
                        "pts": row[26],
                    }
                ]

    seasons = []
    for result_set in career["resultSets"]:
        if result_set["name"] == "SeasonRankingsRegularSeason":
            for row in result_set["rowSet"]:
                for season in season_list[f"{player_id}_{int(row[1].split('-')[0])}"]:
                    seasons += [
                        {
                            **season,
                            **{
                                "rank_min": 1000 if row[8] is None or row[8] == "NR" else row[8],
                                "rank_fgm": 1000 if row[9] is None or row[9] == "NR" else row[9],
                                "rank_fga": 1000 if row[10] is None or row[10] == "NR" else row[10],
                                "rank_fg_pct": 1000 if row[11] is None or row[11] == "NR" else row[11],
                                "rank_fg3m": 1000 if row[12] is None or row[12] == "NR" else row[12],
                                "rank_fg3a": 1000 if row[13] is None or row[13] == "NR" else row[13],
                                "rank_fg3_pct": 1000 if row[14] is None or row[14] == "NR" else row[14],
                                "rank_ftm": 1000 if row[15] is None or row[15] == "NR" else row[15],
                                "rank_fta": 1000 if row[16] is None or row[16] == "NR" else row[16],
                                "rank_ft_pct": 1000 if row[17] is None or row[17] == "NR" else row[17],
                                "rank_oreb": 1000 if row[18] is None or row[18] == "NR" else row[18],
                                "rank_dreb": 1000 if row[19] is None or row[19] == "NR" else row[19],
                                "rank_reb": 1000 if row[20] is None or row[20] == "NR" else row[20],
                                "rank_ast": 1000 if row[21] is None or row[21] == "NR" else row[21],
                                "rank_stl": 1000 if row[22] is None or row[22] == "NR" else row[22],
                                "rank_blk": 1000 if row[23] is None or row[23] == "NR" else row[23],
                                "rank_tov": 1000 if row[24] is None or row[24] == "NR" else row[24],
                                "rank_pts": 1000 if row[25] is None or row[25] == "NR" else row[25],
                                "rank_eff": 1000 if row[26] is None or row[26] == "NR" else row[26],
                            },
                        }
                    ]

    return seasons
