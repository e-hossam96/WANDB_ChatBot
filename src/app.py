"""A Simple chatbot that uses the LangChain and Gradio UI to answer questions about wandb documentation."""
import os
import json
import wandb
import logging
import gradio as gr
from create_chain import load_prompt_template, load_vector_store, load_chain, get_answer
from args import get_chat_parser

os.environ["LANGCHAIN_WANDB_TRACING"] = "true"


class Chat:
    """A chatbot interface that persists the vectorstore and chain between calls."""

    def __init__(
        self,
        args,
    ):
        """Initialize the chatbot.
        Args:
            access_tokens_path: The credentials.
        """
        self.args = args
        with open(self.args.access_tokens_path) as f:
            access_tokens = json.load(f)
        self.access_tokens = access_tokens
        wandb.login(key=self.access_tokens["wandb"]["login"])
        self.wandb_run = wandb.init(
            project="LLMApp",
        )
        self.vector_store = None
        self.chain = None

    def __call__(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None,
        openai_api_key: str = None,
    ):
        """Answer a question about wandb documentation using the LangChain QA chain and vector store retriever.
        Args:
            question (str): The question to answer.
            history (list[tuple[str, str]] | None, optional): The chat history. Defaults to None.
            openai_api_key (str, optional): The OpenAI API key. Defaults to None.
        Returns:
            list[tuple[str, str]], list[tuple[str, str]]: The chat history before and after the question is answered.
        """
        if openai_api_key is not None:
            openai_key = openai_api_key
        elif "openai" in self.access_tokens:
            openai_key = self.access_tokens["openai"]["isemantics"]["hossam"]
        else:
            raise ValueError(
                "Please provide your OpenAI API key as an argument or set the OPENAI_API_KEY environment variable"
            )

        if self.vector_store is None:
            self.vector_store = load_vector_store(
                self.args.vector_db, self.args.access_tokens_path
            )
        prompt_temp = load_prompt_template(self.args.prompt_temp_path)
        if self.chain is None:
            self.chain = load_chain(
                prompt_temp, self.vector_store, self.args.access_tokens_path
            )

        history = history or []
        question = question.lower()
        response = get_answer(
            chain=self.chain,
            question=question,
            chat_history=history,
        )
        history.append((question, response))
        return history, history


with gr.Blocks() as demo:
    gr.HTML(
        """<div style="text-align: center; max-width: 700px; margin: 0 auto;">
        <div
        style="
            display: inline-flex;
            align-items: center;
            gap: 0.8rem;
            font-size: 1.75rem;
        "
        >
        <h1 style="font-weight: 900; margin-bottom: 7px; margin-top: 5px;">
            Wandb QandA Bot
        </h1>
        </div>
        <p style="margin-bottom: 10px; font-size: 94%">
        Hi, I'm a wandb documentaion Q and A bot, start by typing in your OpenAI API key, questions/issues you have related to wandb usage and then press enter.<br>
        Built using <a href="https://langchain.readthedocs.io/en/latest/" target="_blank">LangChain</a> and <a href="https://github.com/gradio-app/gradio" target="_blank">Gradio Github repo</a>
        </p>
    </div>"""
    )
    with gr.Row():
        question = gr.Textbox(
            label="Type in your questions about wandb here and press Enter!",
            placeholder="How do i log images with wandb ?",
        )
        openai_api_key = gr.Textbox(
            type="password",
            label="Enter your OpenAI API key here",
        )
    state = gr.State()
    chatbot = gr.Chatbot()
    question.submit(
        Chat(
            args=get_chat_parser().parse_args(),
        ),
        [question, state, openai_api_key],
        [chatbot, state],
    )


if __name__ == "__main__":
    demo.launch()
