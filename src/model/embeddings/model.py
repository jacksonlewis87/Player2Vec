import torch
from pytorch_lightning import LightningModule
from torch import nn, optim
from typing import Tuple

from model.embeddings.model_config import ModelConfig


class Player2VecModel(LightningModule):
    def __init__(self, config: ModelConfig, loss: nn.Module, stage: str = "train") -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.loss = loss

        self.model = self.init_model(stage=stage)

    def init_model(self, stage: str) -> nn.Sequential:
        model = nn.Sequential()
        model.add_module("embedding_layer", nn.Embedding(self.config.num_embeddings, self.config.embedding_size))

        inference_layer = nn.Linear(self.config.embedding_size, self.config.num_fields)
        if stage == "eval":
            inference_layer = self.load_and_freeze_inference_layer_state_dict(inference_layer=inference_layer)

        model.add_module("inference_layer", inference_layer)
        model.add_module("tanh", nn.Tanh())
        return model

    def load_and_freeze_inference_layer_state_dict(self, inference_layer: nn.Module) -> nn.Module:
        state_dict = torch.load(self.config.checkpoint_path)["state_dict"]

        prefix = "model.inference_layer."
        state_dict = {k.replace(prefix, ""): v for k, v in state_dict.items() if prefix in k}

        inference_layer.load_state_dict(state_dict=state_dict)

        for param in inference_layer.parameters():
            param.requires_grad = False

        return inference_layer

    def forward(self, x: torch.tensor) -> torch.tensor:
        output = self.model(x)
        return output

    def training_step(self, batch: Tuple[torch.tensor, torch.tensor], batch_idx: int) -> float:
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.config.learning_rate)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)

        if self.config.lr_step_size is not None and self.config.lr_gamma is not None:
            scheduler = {
                "scheduler": optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=self.config.lr_step_size,
                    gamma=self.config.lr_gamma,
                ),
            }
            return [optimizer], [scheduler]

        else:
            return optimizer
