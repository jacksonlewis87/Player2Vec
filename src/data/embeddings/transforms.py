from dataclasses import dataclass

from constants import EVAL_SEASONS


def preprocess_field(field: str, value):
    if field == "fg3_pct":
        return max(min(value, 0.6), 0.2)
    elif field == "fg_pct":
        return max(min(value, 0.7), 0.3)
    elif field == "ft_pct":
        return max(value, 0.5)
    elif field[:5] == "rank_":
        return min(value, 600)
    else:
        return value


@dataclass
class FieldNorm:
    min: float = None
    max: float = None


def normalize_dataset(dataset: dict, keys_to_ignore: list):
    fields = list(list(dataset.values())[0].keys())
    for field in keys_to_ignore:
        fields.remove(field)
    fields.sort()

    norms = {field: FieldNorm() for field in fields}

    # calculate mins/maxs
    for sample in dataset.values():
        for field in fields:
            processed_value = preprocess_field(field=field, value=sample[field])
            if norms[field].min is None or processed_value < norms[field].min:
                norms[field].min = processed_value
            if norms[field].max is None or processed_value > norms[field].max:
                norms[field].max = processed_value

    print(norms)

    # normalize
    new_dataset = {}
    for player_id, sample in dataset.items():
        normalized_sample = []
        for field in fields:
            processed_value = preprocess_field(field=field, value=sample[field])
            normalized_sample += [
                round((2.0 * ((processed_value - norms[field].min) / (norms[field].max - norms[field].min))) - 1.0, 3)
            ]
        new_dataset[player_id] = normalized_sample

    return new_dataset


def remove_total_seasons(dataset: list):
    tot_seasons = {}
    for sample in dataset:
        if sample["team_id"] == 0:
            if sample["player_id"] not in tot_seasons:
                tot_seasons[sample["player_id"]] = []

            tot_seasons[sample["player_id"]] += [sample["season"]]

    new_dataset = []
    for sample in dataset:
        if not (
            sample["player_id"] in tot_seasons
            and sample["season"] in tot_seasons[sample["player_id"]]
            and sample["team_id"] != 0
        ):
            new_dataset += [sample]

    return new_dataset


def remove_eval_seasons(dataset: list):
    return [sample for sample in dataset if sample["season"] not in EVAL_SEASONS]
