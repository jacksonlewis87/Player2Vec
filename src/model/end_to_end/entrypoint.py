import os
import torch

from data.end_to_end.data_module import setup_data_module
from model.end_to_end.model import EndToEndGameResultsModel
from model.end_to_end.model_config import FullConfig
from model.model_driver import ModelDriver


def load_state_dict_from_checkpoint(path: str):
    return torch.load(path)["state_dict"]


def train_end_to_end_model(config: FullConfig):
    os.makedirs(config.model_config.experiment_path, exist_ok=True)

    data_module = setup_data_module(config=config.data_config, training_stage=config.model_config.training_stage)
    model = EndToEndGameResultsModel(config=config)

    if config.model_config.state_dict_path:
        print(f"restoring state_dict from {config.model_config.state_dict_path}")
        model.load_state_dict(
            state_dict=load_state_dict_from_checkpoint(config.model_config.state_dict_path), strict=True
        )

    ModelDriver(
        full_config=config,
        model_config=config.model_config,
        model=model,
        data_module=data_module,
    ).run_training()
