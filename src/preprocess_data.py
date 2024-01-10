"""Preprocessing dataset."""

import glob
import json
import logging
from typing import List
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from args import get_processing_parser


def load_data(path: str) -> List[Document]:
    """
    Load dataset from path.

    args:
        - path (str): Data directory

    returns:
        - List[Document]: List of documents
    """
    if path is None:
        raise ValueError("Path to data must be provided.")
    docs = glob.glob(f"{path}/*.md")
    return [UnstructuredMarkdownLoader(d).load()[0] for d in docs]


def chunk_docs(docs: List[Document], chunk_size: str = 500) -> List[Document]:
    """
    Chunk dataset into smaller docs.

    args:
        - docs (List[Document]): List of the raw markdown docs.
        - chunk_size (int): Size of each new document. Defaults to 500.

    returns:
        - List[Document]: Chunked docs.
    """
    splitter = MarkdownTextSplitter(chunk_size=chunk_size)
    return splitter.split_documents(docs)


def create_vector_db(
    docs: List[Document], db_path: str, access_tokens_path: str
) -> None:
    """
    Create vectore store for the docs using Chroma and OpenAI.

    args:
        - docs (List[Document]): List of chunked docs.
        - path (str): Path to save the store
    """
    with open(access_tokens_path) as f:
        openai_api_key = json.load(f)["openai"]["isemantics"]["hossam"]
    encoder = OpenAIEmbeddings(api_key=openai_api_key)
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=encoder,
        persist_directory=db_path,
    )
    vector_db.persist()
    return None


def main(args, logger):
    """Excute preprocessing pipeline."""
    logger.info("Starting W&B_ChatBot Project")
    logger.info("Loading Data")
    docs = load_data(args.docs_dir)
    logger.info("Chunking Docs")
    docs = chunk_docs(docs, args.chunk_size)
    logger.info("Creating Vector Data Base")
    create_vector_db(docs, args.vector_db, args.access_tokens_path)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    args = get_processing_parser().parse_args()
    main(args, logger)
