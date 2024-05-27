from pytorch_lightning import LightningModule
from torch import nn, optim, tensor
from typing import Tuple

from model.game_results.model_config import ModelConfig


class GameResultsModel(LightningModule):
    def __init__(self, config: ModelConfig, loss: nn.Module):
        super().__init__()
        self.config = config
        self.loss = loss

        self.attention = nn.Sequential()
        for i in range(self.config.num_attention_layers):
            self.attention.add_module(
                f"encoder_layer_{i}",
                nn.TransformerEncoderLayer(
                    d_model=self.config.embedding_size,
                    nhead=self.config.num_heads,
                    dropout=self.config.dropout,
                ),
            )

        self.sequential = nn.Sequential()
        self.sequential.add_module("flatten", nn.Flatten())
        self.sequential.add_module(
            "layer_0", nn.Linear(self.config.max_num_players * 2 * self.config.embedding_size, self.config.hidden_dim)
        )
        self.sequential.add_module("dropout_0", nn.Dropout(p=self.config.dropout))
        self.sequential.add_module("relu_0", nn.ReLU())
        self.sequential.add_module("layer_1", nn.Linear(self.config.hidden_dim, 1))
        self.sequential.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (sequence_length, batch_size, input_dim)
        x = self.attention(x)
        x = x.permute(1, 0, 2)  # (batch_size, sequence_length, input_dim)
        return self.sequential(x)

    def training_step(self, batch: Tuple[tensor, tensor], batch_idx: int) -> float:
        x, y, _ = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Tuple[tensor, tensor], batch_idx: int) -> None:
        x, y, _ = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)
        scheduler = {
            "scheduler": optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.learning_rate,
                total_steps=self.config.epochs,
            ),
        }
        return [optimizer], [scheduler]
