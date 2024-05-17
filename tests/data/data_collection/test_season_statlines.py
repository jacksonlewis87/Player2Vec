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
