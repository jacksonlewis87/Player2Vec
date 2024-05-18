import pytest
from unittest.mock import patch, call

from data.data_collection.games import get_players_for_game, get_player_ids, get_season_game_results


@pytest.mark.parametrize("filter_home_games", [False, True])
@patch("data.data_collection.games.leaguegamelog.LeagueGameLog")
@patch("data.data_collection.games.row_to_header_dict")
@patch("data.data_collection.games.filter_dict_fields")
def test_get_season_game_results(
    mock_filter_dict_fields, mock_row_to_header_dict, mock_league_game_log, filter_home_games
):
    season = 2020
    headers = "test_headers"
    game_0 = {"field_0": "value_0", "MATCHUP": "test vs. test"}
    game_1 = {"field_1": "value_1", "MATCHUP": "test"}
    fields_to_include = ["GAME_ID", "WL"]
    mock_league_game_log.return_value.get_dict.return_value = {"resultSets": [{"headers": headers, "rowSet": [0, 1]}]}
    mock_row_to_header_dict.side_effect = [game_0, game_1]

    result = get_season_game_results(season=season, filter_home_games=filter_home_games)

    mock_league_game_log.assert_called_once_with(season=season, league_id="00", season_type_all_star="Regular Season")
    mock_league_game_log.return_value.get_dict.assert_called_once()
    mock_row_to_header_dict.assert_has_calls([call(headers=headers, row=0), call(headers=headers, row=1)])

    expected_call_list = [call(game_0, fields_to_include=fields_to_include)]
    expected_result_list = [mock_filter_dict_fields.return_value]
    if not filter_home_games:
        expected_call_list += [call(game_1, fields_to_include=fields_to_include)]
        expected_result_list += [mock_filter_dict_fields.return_value]

    mock_filter_dict_fields.assert_has_calls(expected_call_list)
    assert result == expected_result_list


def test_get_player_ids():
    row = [{"personId": 1, "test": None}, {"personId": "foo", "nope": 2}]

    result = get_player_ids(row=row)

    assert result == [1, "foo"]


@patch("data.data_collection.games.boxscoretraditionalv3.BoxScoreTraditionalV3")
@patch("data.data_collection.games.get_player_ids")
def test_get_players_for_game(mock_get_player_ids, mock_box_score_traditional_v3):
    game_id = "game_id"
    mock_box_score_traditional_v3.return_value.get_dict.return_value = {
        "boxScoreTraditional": {
            "awayTeam": {"players": "away_team_ids"},
            "homeTeam": {"players": "home_team_ids"},
        }
    }

    result = get_players_for_game(game_id=game_id)

    mock_box_score_traditional_v3.assert_called_once_with(game_id=game_id)
    mock_get_player_ids.assert_has_calls([call("away_team_ids"), call("home_team_ids")])
    assert result[0] == mock_get_player_ids.return_value
    assert result[1] == mock_get_player_ids.return_value
