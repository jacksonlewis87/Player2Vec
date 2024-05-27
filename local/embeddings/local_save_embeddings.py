from constants import ROOT_DIR
from model.embeddings.extract_embeddings import extract_game_embeddings


def do_work(path_to_checkpoint: str, path_to_encodings: str, output_path: str):
    extract_game_embeddings(
        path_to_checkpoint=path_to_checkpoint,
        path_to_encodings=path_to_encodings,
        output_path=output_path,
    )


if __name__ == "__main__":
    embedding_version = 1
    version = 0
    epoch = 499
    step = 48500
    output_path_base = f"{ROOT_DIR}/data"
    checkpoint_path = f"{ROOT_DIR}/data/training/embeddings_v{embedding_version}/lightning_logs/version_{version}/checkpoints/epoch={epoch}-step={step}.ckpt"
    encoding_path = f"{ROOT_DIR}/data/training/embeddings_v{embedding_version}"
    output_file_name = f"embeddings_v{embedding_version}-{version}.json"

    # eval
    eval_version = 7
    eval_epoch = 999
    eval_step = 10000
    checkpoint_path = f"{output_path_base}/training/embeddings_v{embedding_version}/lightning_logs/version_{version}/epoch={epoch}-step={step}/lightning_logs/version_{eval_version}/checkpoints/epoch={eval_epoch}-step={eval_step}.ckpt"
    encoding_path = f"{output_path_base}/training/embeddings_v{embedding_version}/lightning_logs/version_{version}/epoch={epoch}-step={step}"
    output_file_name = f"eval_{output_file_name}"

    do_work(
        path_to_checkpoint=checkpoint_path,
        path_to_encodings=encoding_path,
        output_path=f"{ROOT_DIR}/data/{output_file_name}",
    )
