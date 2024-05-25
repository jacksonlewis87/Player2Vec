import pytest
from unittest.mock import patch

from data.embeddings.transforms import (
    preprocess_field,
    normalize_dataset,
    remove_non_total_seasons,
    remove_eval_seasons,
    remove_eval_seasons_game_id,
    convert_to_list,
)


@pytest.mark.parametrize(
    "field, value, expected_result",
    [
        ("fg3_pct", 0.7, 0.6),
        ("fg3_pct", 0.5, 0.5),
        ("fg3_pct", 0.1, 0.2),
        ("fg_pct", 0.8, 0.7),
        ("fg_pct", 0.5, 0.5),
        ("fg_pct", 0.2, 0.3),
        ("ft_pct", 0.7, 0.7),
        ("ft_pct", 0.4, 0.5),
        ("rank_", 700, 600),
        ("rank_", 500, 500),
        ("test", 1, 1),
    ],
)
def test_preprocess_field(field, value, expected_result):
    result = preprocess_field(field=field, value=value)

    assert result == expected_result


@patch("data.embeddings.transforms.preprocess_field")
def test_normalize_dataset(mock_prep_process_field):
    mock_prep_process_field.side_effect = lambda field, value: value
    dataset = {
        "sample_0": {
            "field_0": 0,
            "field_1": 1,
            "field_2": 2,
        },
        "sample_1": {
            "field_0": 3,
            "field_1": 4,
            "field_2": 5,
        },
        "sample_2": {
            "field_0": 6,
            "field_1": 7,
            "field_2": 8,
        },
    }

    result = normalize_dataset(dataset=dataset, keys_to_ignore=["field_1"])

    assert result == {"sample_0": [-1.0, -1.0], "sample_1": [0.0, 0.0], "sample_2": [1.0, 1.0]}


def test_remove_total_seasons():
    player_season_0 = {"team_id": 1, "player_id": 0, "season": 0}
    player_season_1 = {"team_id": 0, "player_id": 0, "season": 0}
    player_season_2 = {"team_id": 1, "player_id": 0, "season": 1}
    player_season_3 = {"team_id": 1, "player_id": 1, "season": 0}
    dataset = [
        player_season_0,
        player_season_1,
        player_season_2,
        player_season_3,
    ]

    result = remove_non_total_seasons(dataset=dataset)

    assert result == [player_season_1, player_season_2, player_season_3]


@patch("data.embeddings.transforms.EVAL_SEASONS", [1])
def test_remove_eval_seasons():
    sample_0 = {"sample": 0, "season": 1}
    sample_1 = {"sample": 1, "season": 2}
    dataset = [sample_0, sample_1]

    result = remove_eval_seasons(dataset=dataset)

    assert result == [sample_1]


@patch("data.embeddings.transforms.EVAL_SEASONS", [2009])
def test_remove_eval_seasons_game_id():
    sample_0 = {"sample": 0, "season": 1}
    sample_1 = {"sample": 1, "season": 2}
    dataset = {
        "0020900": sample_0,
        "0020800": sample_1,
    }

    result = remove_eval_seasons_game_id(dataset=dataset)

    assert result == {"0020800": sample_1}


def test_convert_to_list():
    sample_0 = {"sample": 0, "season": 1}
    sample_1 = {"sample": 1, "season": 2}
    dataset = {"val_0": {"val_1": sample_0, "val_2": sample_1}}

    result = convert_to_list(dataset=dataset, key_field_names=["field_0", "field_1"])

    assert result == [
        {**{"field_0": "val_0", "field_1": "val_1"}, **sample_0},
        {**{"field_0": "val_0", "field_1": "val_2"}, **sample_1},
    ]
