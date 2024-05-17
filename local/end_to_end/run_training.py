from constants import ROOT_DIR, EndToEndStage
from data.end_to_end.data_config import DataConfig
from model.end_to_end.model_config import FullConfig, ModelConfig
from model.end_to_end.entrypoint import train_end_to_end_model


def do_work():
    experiment_name = "end_to_end_v0"

    stage = EndToEndStage.END_TO_END.value
    checkpoint_version = 0
    checkpoint_epoch = 149
    checkpoint_step = 115050

    config = FullConfig(
        data_config=DataConfig(
            embeddings_path=f"{ROOT_DIR}/data/embeddings_v0-14.json",
            embedding_data_path=f"{ROOT_DIR}/data/data_profile_2000.json",
            game_results_path=f"{ROOT_DIR}/data/game_results.json",
            data_split_path=f"{ROOT_DIR}/data/training/{experiment_name}/data_split.json",
            batch_size=16,
            train_size=0.8,
            shuffle_players=True,
            pad_team_players=15,
            num_embeddings=13971,
            embedding_inference_keys_to_ignore=["player_id", "season", "team_id"],
        ),
        model_config=ModelConfig(
            experiment_path=f"{ROOT_DIR}/data/training/{experiment_name}",
            learning_rate=0.00003,
            epochs=200,
            checkpoint_path=None,
            state_dict_path=None,
            # state_dict_path=f"{ROOT_DIR}/data/training/{experiment_name}/lightning_logs/version_{checkpoint_version}/checkpoints/epoch={checkpoint_epoch}-step={checkpoint_step}.ckpt",
            embedding_size=12,
            num_fields=41,
            hidden_dim=16,
            num_heads=4,
            num_attention_layers=2,
            dropout=0.4,
            training_stage=stage,
            # embedding_loss_weight=2.0,
            embedding_loss_weight=0.0,
        ),
    )

    train_end_to_end_model(config=config)


if __name__ == "__main__":
    do_work()
