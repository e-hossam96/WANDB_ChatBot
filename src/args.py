"""Argument parsers for project."""

import argparse


def get_base_parser():
    """Create base argparser shared by all calls."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vector_db",
        type=str,
        default="./data/vector_db/",
        help="The directory to save or load the Chroma db to/from",
    )
    parser.add_argument(
        "--access_tokens_path",
        type=str,
        default="./data/access_tokens.json",
        help="The directory to access credentials",
    )
    return parser


def get_processing_parser():
    """Create argparser for data preprocessing."""
    parser = get_base_parser()
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
    return parser


def get_chat_parser():
    """Create argparser for convesation."""
    parser = get_base_parser()
    parser.add_argument(
        "--prompt_temp_path",
        type=str,
        default="./data/prompt_template.json",
        help="The directory to prompt templates.",
    )
    return parser
