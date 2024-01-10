"""Argument parsers for project."""

import argparse


def get_processing_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs_dir",
        type=str,
        required=True,
        help="The directory containing the wandb documentation",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="The number of tokens to include in each document chunk",
    )
    parser.add_argument(
        "--vector_store",
        type=str,
        default="./data/vector_store/",
        help="The directory to save or load the Chroma db to/from",
    )

    return parser
