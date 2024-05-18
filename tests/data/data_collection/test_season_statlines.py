from unittest.mock import patch

from data.data_collection.season_statlines import get_all_players, get_player_profile, Player


@patch("data.data_collection.season_statlines.commonallplayers.CommonAllPlayers")
def test_get_all_players(mock_common_all_players):
    player_0 = Player(id=0, name="name_0", roster_status=1, start_year=2, end_year=3)
    player_1 = Player(id=4, name="name_1", roster_status=5, start_year=6, end_year=7)
    mock_common_all_players.return_value.get_dict.return_value = {
        "resultSets": [
            {
                "rowSet": [
                    [player_0.id, None, player_0.name, player_0.roster_status, player_0.start_year, player_0.end_year],
                    [player_1.id, None, player_1.name, player_1.roster_status, player_1.start_year, player_1.end_year],
                ]
            }
        ]
    }

    result = get_all_players()

    mock_common_all_players.assert_called_once()
    mock_common_all_players.return_value.get_dict.assert_called_once()
    assert result == [player_0, player_1]


@patch("data.data_collection.season_statlines.playerprofilev2.PlayerProfileV2")
def test_get_player_profile(mock_player_profile_v2):
    player_id = 2
    mock_player_profile_v2.return_value.get_dict.return_value = {
        "resultSets": [
            {"name": "SeasonTotalsRegularSeason", "rowSet": [[None, "2024-season"] + [0 for _ in range(100)]]},
            {
                "name": "SeasonRankingsRegularSeason",
                "rowSet": [[None, "2024-season"] + [1 for _ in range(24)] + ["NR"]],
            },
        ]
    }

    result = get_player_profile(player_id=player_id)

    mock_player_profile_v2.assert_called_once_with(player_id=player_id)
    mock_player_profile_v2.return_value.get_dict.assert_called_once()
    assert result[0]["player_id"] == player_id
    assert result[0]["season"] == 2024
    assert result[0]["team_id"] == 0
    assert result[0]["pts"] == 0
    assert result[0]["rank_min"] == 1
    assert result[0]["rank_pts"] == 1
    assert result[0]["rank_eff"] == 1000
