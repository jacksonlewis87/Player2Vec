from constants import ROOT_DIR
from data.embeddings.data_config import DataConfig
from loss.losses import Loss
from model.embeddings.model_config import FullConfig, ModelConfig
from model.embeddings.entrypoint import train_embeddings_model


def do_work():
    experiment_name = "embeddings_v1"

    checkpoint_version = 0
    checkpoint_epoch = 499
    checkpoint_step = 48500

    config = FullConfig(
        data_config=DataConfig(
            data_path=f"{ROOT_DIR}/data/recent_game_stats_30.json",
            batch_size=4096,
            keys_to_ignore=["player_id", "game_id"],
        ),
        model_config=ModelConfig(
            experiment_path=f"{ROOT_DIR}\\data\\training\\{experiment_name}\\lightning_logs\\version_{checkpoint_version}\\epoch={checkpoint_epoch}-step={checkpoint_step}",
            learning_rate=0.01,
            lr_step_size=None,
            lr_gamma=None,
            # learning_rate=0.03,
            # lr_step_size=25,
            # lr_gamma=.9,
            epochs=1000,
            checkpoint_path=f"{ROOT_DIR}\\data\\training\\{experiment_name}\\lightning_logs\\version_{checkpoint_version}\\checkpoints\\epoch={checkpoint_epoch}-step={checkpoint_step}.ckpt",
            loss=Loss.L2.value,
            embedding_size=8,
            num_embeddings=37158,
            num_fields=21,
        ),
    )

    train_embeddings_model(config=config, stage="eval")


if __name__ == "__main__":
    do_work()
