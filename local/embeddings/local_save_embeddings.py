from constants import ROOT_DIR
from model.embeddings.extract_embeddings import extract_game_embeddings


def do_work(embedding_version: int, version: int, epoch: int, step: int, output_path: str):
    extract_game_embeddings(
        path_to_checkpoint=f"{ROOT_DIR}/data/training/embeddings_v{embedding_version}/lightning_logs/version_{version}/checkpoints/epoch={epoch}-step={step}.ckpt",
        path_to_encodings=f"{ROOT_DIR}/data/training/embeddings_v{embedding_version}",
        output_path=output_path,
    )


if __name__ == "__main__":
    embedding_version = 1
    version = 0
    epoch = 499
    step = 48500
    output_path = f"{ROOT_DIR}/data/embeddings_v{embedding_version}-{version}.json"

    do_work(
        embedding_version=embedding_version,
        version=version,
        epoch=epoch,
        step=step,
        output_path=output_path,
    )
