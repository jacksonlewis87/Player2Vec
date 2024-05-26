import os
from enum import Enum


class EndToEndStage(Enum):
    EMBEDDING = "embedding"
    GAME_RESULTS = "game_results"
    END_TO_END = "end_to_end"


ROOT_DIR = os.path.abspath(os.path.dirname(__file__)).split("\\src")[0]
EVAL_SEASONS = [2022, 2023]
BAD_GAME_IDS = [
    "0020600608",
    "0020800840",
    "0022100144",
    "0020500057",
    "0020200557",
    "0022100038",
    "0020900786",
    "0022000718",
    "0022000136",
]
