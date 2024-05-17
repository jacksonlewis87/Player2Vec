import torch

from data.game_results.data_module import setup_data_module
from model.game_results.model import GameResultsModel
from model.game_results.model_config import FullConfig


def eval_game_results_model(config: FullConfig):
    data_module = setup_data_module(config=config.data_config, stage="eval")
    model = GameResultsModel.load_from_checkpoint(
        checkpoint_path=config.model_config.checkpoint_path, config=config.model_config, loss=None
    )
    model.eval()

    with torch.no_grad():
        for batch in data_module.val_dataloader():
            x, y, game_ids = batch
            inference = model.forward(x)

            for i in range(y.size(dim=0)):
                print(f"{game_ids[i]}\t{y[i][0].item()}\t{round(inference[i][0].item(), 3)}")
