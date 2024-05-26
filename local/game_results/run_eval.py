from constants import ROOT_DIR
from data.game_results.data_config import DataConfig
from loss.losses import Loss
from model.game_results.model_config import FullConfig, ModelConfig
from model.game_results.eval import eval_game_results_model


def do_work():
    experiment_name = "game_results_v1"
    version = 0
    epoch = 99
    step = 7900

    config = FullConfig(
        data_config=DataConfig(
            game_results_path=f"{ROOT_DIR}/data/game_results.json",
            embeddings_path=f"{ROOT_DIR}/data/embeddings_v1-0.json",
            data_split_path=f"{ROOT_DIR}/data/training/{experiment_name}/data_split.json",
            batch_size=16,
            train_size=0.8,
            shuffle_players=True,
            pad_team_players=15,
        ),
        model_config=ModelConfig(
            experiment_path=f"{ROOT_DIR}/data/training/{experiment_name}",
            learning_rate=0.00003,
            epochs=100,
            checkpoint_path=f"{ROOT_DIR}/data/training/{experiment_name}/lightning_logs/version_{version}/checkpoints/epoch={epoch}-step={step}.ckpt",
            loss=Loss.BCE.value,
            embedding_size=8,
            max_num_players=15,
            hidden_dim=16,
            num_heads=4,
            num_attention_layers=2,
            dropout=0.4,
        ),
    )

    eval_game_results_model(config=config)


if __name__ == "__main__":
    do_work()
