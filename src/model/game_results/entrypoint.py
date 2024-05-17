import os

from data.game_results.data_module import setup_data_module
from loss.losses import get_loss
from model.game_results.model import GameResultsModel
from model.game_results.model_config import FullConfig
from model.model_driver import ModelDriver


def train_game_results_model(config: FullConfig):
    os.makedirs(config.model_config.experiment_path, exist_ok=True)

    data_module = setup_data_module(config=config.data_config)
    loss = get_loss(loss=config.model_config.loss)
    model = GameResultsModel(config=config.model_config, loss=loss)

    ModelDriver(
        full_config=config,
        model_config=config.model_config,
        model=model,
        data_module=data_module,
    ).run_training()
