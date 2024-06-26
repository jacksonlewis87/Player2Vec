from constants import ROOT_DIR
from data.embeddings.data_config import DataConfig
from loss.losses import Loss
from model.embeddings.model_config import FullConfig, ModelConfig
from model.embeddings.entrypoint import train_embeddings_model


def do_work():
    experiment_name = "embeddings_v1"

    checkpoint_version = 1
    checkpoint_epoch = 1
    checkpoint_step = 24726

    config = FullConfig(
        data_config=DataConfig(
            data_path=f"{ROOT_DIR}/data/recent_game_stats_30.json",
            batch_size=4096,
            keys_to_ignore=["player_id", "game_id"],
        ),
        model_config=ModelConfig(
            experiment_path=f"{ROOT_DIR}\\data\\training\\{experiment_name}",
            learning_rate=0.001,
            lr_step_size=None,
            lr_gamma=None,
            epochs=500,
            checkpoint_path=None,
            # checkpoint_path=f"{ROOT_DIR}\\data\\training\\{experiment_name}\\lightning_logs\\version_{checkpoint_version}\\checkpoints\\epoch={checkpoint_epoch}-step={checkpoint_step}.ckpt",
            loss=Loss.L2.value,
            embedding_size=8,
            num_embeddings=395605,
            num_fields=21,
        ),
    )

    train_embeddings_model(config=config)


if __name__ == "__main__":
    do_work()
