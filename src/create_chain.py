"""Defining the conversation chain."""

import json
import wandb
import logging
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from typing import List
from args import get_chat_parser


def load_prompt_template(path: str, index=0) -> ChatPromptTemplate:
    """Load prompt template from JSON file and define it in LC.

    args:
        - path (str): Path to prompt templates file.
        - index (int): Index of the to-be-used template. Defaults to 0.

    returns:
        - prompt_temp (ChatPromptTemplate): Prompt template.
    """
    with open(path, encoding="utf-8") as f:
        temps = json.load(f)
    temp = temps[index]
    messages = [
        SystemMessagePromptTemplate.from_template(temp["system_template"]),
        HumanMessagePromptTemplate.from_template(temp["human_template"]),
    ]
    prompt_temp = ChatPromptTemplate.from_messages(messages)
    return prompt_temp


def load_vector_store(db_path, access_tokens_path):
    """Load vector store from local.

    args:
        - db_path (srt): Path to vector storage.
        - access_tokens_path (str): Path to OpenAI keys.

    returns:
        - Chroma: A chroma vector store object.
    """
    with open(access_tokens_path) as f:
        openai_api_key = json.load(f)["openai"]["isemantics"]["hossam"]
    encoder = OpenAIEmbeddings(api_key=openai_api_key)
    vector_db = Chroma(
        embedding_function=encoder,
        persist_directory=db_path,
    )
    return vector_db


def load_chain(prompt_temp, vector_db, access_tokens_path):
    """Load the conversational chain.

    args:
        - prompt_temp (ChatPromptTemplate): The chat template from LC.
        - vector_db (Chroma): The vector storage.
        - access_tokens_path (str): Path to OpenAI keys.

    returns:
        - ConversationalRetrievalChain: A ConversationalRetrievalChain object.
    """
    with open(access_tokens_path) as f:
        openai_api_key = json.load(f)["openai"]["isemantics"]["hossam"]
    retriever = vector_db.as_retriever()
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo",
        temperature=0.1,
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt_temp},
        return_source_documents=True,
    )
    return qa_chain


def get_answer(
    chain: ConversationalRetrievalChain,
    question: str,
    chat_history: List[tuple[str, str]],
) -> str:
    """Get answer from conversational object.

    args:
        - chain (ConversationalRetrievalChain): A ConversationalRetrievalChain object.
        - question (str): A query.
        - chat_history (list): Past conversations.

    returns:
        - str: The answer.
    """
    result = chain(
        inputs={"question": question, "chat_history": chat_history},
        return_only_outputs=True,
    )
    response = f"Answer:\t{result['answer']}"
    return response
