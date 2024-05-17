from dataclasses import dataclass


@dataclass
class DataConfig:
    data_path: str
    batch_size: int
    keys_to_ignore: list[str]
