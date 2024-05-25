from pytorch_lightning import LightningModule
from torch import nn, optim, tensor
from typing import Tuple

from model.embeddings.model_config import ModelConfig


class Player2VecModel(LightningModule):
    def __init__(
        self,
        config: ModelConfig,
        loss: nn.Module,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.loss = loss

        self.model = self.init_model()

    def init_model(self) -> nn.Sequential:
        model = nn.Sequential()
        model.add_module("embedding_layer", nn.Embedding(self.config.num_embeddings, self.config.embedding_size))
        model.add_module("inference_layer", nn.Linear(self.config.embedding_size, self.config.num_fields))
        model.add_module("tanh", nn.Tanh())
        return model

    def forward(self, x: tensor) -> tensor:
        output = self.model(x)
        return output

    def training_step(self, batch: Tuple[tensor, tensor], batch_idx: int) -> float:
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.config.learning_rate)
