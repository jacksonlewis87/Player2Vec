from data.embeddings.data_module import setup_data_module
from loss.losses import get_loss
from model.embeddings.model import Player2VecModel
from model.embeddings.model_config import FullConfig
from model.model_driver import ModelDriver


def train_embeddings_model(config: FullConfig, stage: str = "train"):
    data_module = setup_data_module(config=config, stage=stage)
    loss = get_loss(loss=config.model_config.loss)
    model = Player2VecModel(config=config.model_config, loss=loss, stage=stage)

    if stage == "eval":
        config.model_config.checkpoint_path = None

    ModelDriver(
        full_config=config,
        model_config=config.model_config,
        model=model,
        data_module=data_module,
    ).run_training()
