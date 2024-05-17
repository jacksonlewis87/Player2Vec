import os
from enum import Enum


class EndToEndStage(Enum):
    EMBEDDING = "embedding"
    GAME_RESULTS = "game_results"
    END_TO_END = "end_to_end"


ROOT_DIR = os.path.abspath(os.path.dirname(__file__)).split("\\src")[0]
EVAL_SEASONS = [2022, 2023]
BAD_GAME_IDS = [
    "0020900885",
    "0020300428",
]
