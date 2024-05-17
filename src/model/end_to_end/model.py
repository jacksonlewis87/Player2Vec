import json
from collections import OrderedDict
from pytorch_lightning import LightningModule
from torch import nn, optim, tensor, ones, zeros, stack, cat
from typing import Tuple

from constants import EndToEndStage
from data.end_to_end.data_module import EmbeddingDataset
from loss.losses import BCELoss, L2Loss, WeightedL2Loss
from model.end_to_end.model_config import FullConfig


class EndToEndGameResultsModel(LightningModule):
    def __init__(self, config: FullConfig):
        super().__init__()
        self.config = config
        self.bce_loss = BCELoss()
        self.l2_loss = L2Loss()
        self.weighted_l2_loss = WeightedL2Loss()
        self.embedding_layer = None
        self.learnable_embedding_layer = None
        self.embedding_inference_layer = None
        self.embedding_inference_loss_weights = None
        self.game_results_attention = None
        self.game_results_sequential = None
        self.num_learnable_embeddings = 5000
        self.learnable_embeddings_counter = 0
        self.learnable_embeddings_map = {}

        if self.config.model_config.training_stage != EndToEndStage.EMBEDDING.value:
            self.embedding_dataset = EmbeddingDataset(config=self.config.data_config)

        # self.init_embedding_layer()
        self.init_learnable_embedding_layer()
        self.init_embedding_inference_layer()
        self.init_embedding_inference_loss_weights()
        self.init_game_results_model()

        self.cache_embeddings = None
        self.cached_embeddings = {}

    # def init_embedding_layer(self):
    #     self.embedding_layer = nn.Sequential(OrderedDict([
    #         ("embedding_layer", nn.Linear(self.config.data_config.num_embeddings, self.config.model_config.embedding_size, bias=False))
    #     ]))
    #
    #     if self.config.model_config.training_stage == EndToEndStage.GAME_RESULTS.value:
    #         for param_name, parameter in self.embedding_layer.named_parameters():
    #             parameter.requires_grad = False

    def init_learnable_embedding_layer(self):
        self.learnable_embedding_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "learnable_embedding_layer",
                        nn.Linear(self.num_learnable_embeddings, self.config.model_config.embedding_size, bias=False),
                    )
                ]
            )
        )

    def init_embedding_inference_layer(self):
        self.embedding_inference_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "embedding_inference_layer",
                        nn.Linear(self.config.model_config.embedding_size, self.config.model_config.num_fields),
                    ),
                    ("embedding_inference_tanh", nn.Tanh()),
                ]
            )
        )

        if self.config.model_config.training_stage == EndToEndStage.GAME_RESULTS.value:
            for param_name, parameter in self.embedding_inference_layer.named_parameters():
                parameter.requires_grad = False

    def init_embedding_inference_loss_weights(self):
        self.embedding_inference_loss_weights = nn.Parameter(
            ones(self.config.model_config.num_fields),
            requires_grad=False if self.config.model_config.training_stage != EndToEndStage.END_TO_END.value else True,
        )

    def init_game_results_model(self):
        self.game_results_attention = [
            nn.TransformerEncoderLayer(
                d_model=self.config.model_config.embedding_size,
                nhead=self.config.model_config.num_heads,
                dropout=self.config.model_config.dropout,
            )
            for _ in range(self.config.model_config.num_attention_layers)
        ]

        self.game_results_sequential = nn.Sequential(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    (
                        "layer_0",
                        nn.Linear(
                            self.config.data_config.pad_team_players * 2 * self.config.model_config.embedding_size,
                            self.config.model_config.hidden_dim,
                        ),
                    ),
                    ("dropout_0", nn.Dropout(p=self.config.model_config.dropout)),
                    ("relu_0", nn.ReLU()),
                    ("layer_1", nn.Linear(self.config.model_config.hidden_dim, 1)),
                    ("sigmoid", nn.Sigmoid()),
                ]
            )
        )

        if self.config.model_config.training_stage == EndToEndStage.EMBEDDING.value:
            for attention_layer in self.game_results_attention:
                for param_name, parameter in attention_layer.named_parameters():
                    parameter.requires_grad = False
            for param_name, parameter in self.game_results_sequential.named_parameters():
                parameter.requires_grad = False

    def register_learnable_embedding(self, player_id: str):
        if self.learnable_embeddings_counter >= self.num_learnable_embeddings:
            print("Error: too many learnable embeddings")
            raise Exception
        else:
            self.learnable_embeddings_map[player_id] = self.learnable_embeddings_counter
            self.learnable_embeddings_counter += 1

    def get_learnable_embedding_encoding(self, player_id: str):
        if player_id not in self.learnable_embeddings_map:
            self.register_learnable_embedding(player_id=player_id)
        encoding_number = self.learnable_embeddings_map[player_id]
        return tensor([1.0 if i == encoding_number else 0.0 for i in range(self.num_learnable_embeddings)])

    def forward_learnable_embeddings(self, player_id_list: list[str]):
        encodings = stack(
            [self.get_learnable_embedding_encoding(player_id=player_id) for player_id in player_id_list], dim=0
        )
        return self.learnable_embedding_layer(encodings)

    def forward_embedding(self, x):
        return self.embedding_layer(x)

    def forward_embedding_inference(self, x):
        return self.embedding_inference_layer(x)

    def forward_game_results(self, x):
        x = x.permute(1, 0, 2)  # (sequence_length, batch_size, input_dim)
        for attention_layer in self.game_results_attention:
            x = attention_layer(x)
        x = x.permute(1, 0, 2)  # (batch_size, sequence_length, input_dim)
        return self.game_results_sequential(x)

    def forward(self, x):
        if (
            self.config.model_config.training_stage == EndToEndStage.EMBEDDING.value
            or self.config.model_config.training_stage == EndToEndStage.END_TO_END.value
        ):
            x = self.forward_embedding(x)
        if self.config.model_config.training_stage == EndToEndStage.EMBEDDING.value:
            x = self.forward_embedding_inference(x)
        if (
            self.config.model_config.training_stage == EndToEndStage.GAME_RESULTS.value
            or self.config.model_config.training_stage == EndToEndStage.END_TO_END.value
        ):
            x = self.forward_game_results(x)
        return x

    def training_step(self, batch: Tuple, batch_idx: int) -> float:
        if self.config.model_config.training_stage == EndToEndStage.EMBEDDING.value:
            return self.embedding_step(batch=batch, stage="train")
        elif self.config.model_config.training_stage == EndToEndStage.GAME_RESULTS.value:
            # after 1 epoch switch to cached embeddings
            if self.cache_embeddings is not None:
                self.cache_embeddings = True

            return self.game_results_step(batch=batch, stage="train")
        else:
            return self.end_to_end_step(batch=batch, stage="train")

    def validation_step(self, batch: Tuple, batch_idx: int):
        if self.config.model_config.training_stage == EndToEndStage.EMBEDDING.value:
            return None
        elif self.config.model_config.training_stage == EndToEndStage.GAME_RESULTS.value:
            # skipping sanity validation, let model know 1 epoch is done
            if (
                self.cache_embeddings is None
                and self.config.model_config.training_stage == EndToEndStage.GAME_RESULTS.value
                and not self.cache_embeddings
                and batch_idx > 2
            ):
                self.cache_embeddings = False
            return self.game_results_step(batch=batch, stage="val")
        else:
            return self.end_to_end_step(batch=batch, stage="val")

    def embedding_step(self, batch: Tuple, stage: str):
        x, y, _ = batch
        y_pred = self.forward(x)
        loss = self.l2_loss(y_pred, y)
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def game_results_step(self, batch: Tuple, stage: str):
        x, y, _ = batch

        y_pred = self.forward_game_results(x)
        loss = self.bce_loss(y_pred, y)
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def process_learnable_embeddings_input(self, x: tensor, l_e: list):
        for b in range(x.size(dim=0)):
            player_id_list = [e for e in l_e[b] if e != ""]
            if len(player_id_list) > 0:
                embeddings = self.forward_learnable_embeddings(player_id_list=player_id_list)
                embedding_indexes = [i for i, e in enumerate(l_e[b]) if e != ""]
                for i in range(len(embedding_indexes)):
                    x[b][embedding_indexes[i]] = embeddings[i]
        return x

    def end_to_end_step(self, batch: Tuple, stage: str):
        x, y, l_e, e_x, e_y, _ = batch

        e_y_pred = self.forward_embedding_inference(e_x)

        x = self.process_learnable_embeddings_input(x=x, l_e=l_e)
        y_pred = self.forward_game_results(x)

        bce_loss = self.bce_loss(y_pred, y)
        l2_loss = self.weighted_l2_loss(
            output=e_y_pred,
            labels=e_y,
            weights=self.embedding_inference_loss_weights,
        )
        loss = bce_loss + (self.config.model_config.embedding_loss_weight * l2_loss)
        self.log(f"{stage}_bce_loss", bce_loss, on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{stage}_l2_loss", l2_loss, on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self) -> None:
        with open(f"{self.config.model_config.experiment_path}/learnable_embeddings_map.json", "w") as f:
            json.dump(self.learnable_embeddings_map, fp=f)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.config.model_config.learning_rate)
