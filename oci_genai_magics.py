"""
This module implements magic commands for integrating an OCI GenAI model
within a Jupyter Notebook.

partially inspired by: 
    https://github.com/vinayak-mehta/ipychat
"""

import logging
from time import time
from IPython.core.magic import Magics, line_magic, magics_class
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import tiktoken

from oci_models import get_llm
from context import filter_variables, get_context
from prompts import PROMPT_ASK, PROMPT_ASK_CODE, PROMPT_ASK_DATA

from config import (
    MODEL_ID,
    SERVICE_ENDPOINT,
    MAX_TOKENS,
    TEMPERATURE,
    MAX_MSGS_IN_HISTORY,
    TOKENIZER,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@magics_class
class OCIGenaiMagics(Magics):
    """
    A class to implement custom magic commands for interacting with an OCI GenAI model.
    """

    def __init__(self, shell):
        """
        Initialize a new instance of OCIGenaiMagics.
        """
        super().__init__(shell)

        # the tokenizer
        self.tokenizer = tiktoken.get_encoding(TOKENIZER)

        # the list of messages
        self.history = []
        # to compute tokens for input + output
        self.tokens_input = 0
        self.tokens_output = 0
        # to compute genai resp.time
        self.genai_requests = 0
        self.genai_total_time = 0

    def compute_tokens(self, messages):
        """
        Compute the #of tokens for the messages.
        """
        total_tokens = 0
        for message in messages:
            # Estrarre il content dal messaggio, gestendo i diversi tipi di messaggi
            if hasattr(message, "content"):
                content = message.content  # SystemMessage, HumanMessage, AIMessage
            else:
                content = str(message)

            if content is None:
                content = ""

            tokens = self.tokenizer.encode(content)
            total_tokens += len(tokens)

        return total_tokens

    def print_stream(self, _ai_response):
        """
        Helper function to print streaming responses from the AI model.

        Args:
            ai_response (generator): A generator yielding chunks of the AI response.

        Returns:
            str: The complete response as a single string.
        """
        all_chunks = ""

        for chunk in _ai_response:
            print(chunk.content, end="", flush=True)
            all_chunks += chunk.content

        # return the entire result to be stored in history
        return all_chunks

    def handle_input(self, messages, last_request):
        """
        Process user input and send it to the AI model.

        Args:
            messages (list): A list of message objects to send to the AI model.
            last_request (str): The user's latest input.

        Returns:
            None
        """
        llm = get_llm()

        time_start = time()

        ai_response = llm.stream(messages)

        all_text = self.print_stream(ai_response)

        # update stats
        self.genai_requests += 1
        self.genai_total_time += time() - time_start
        self.tokens_input += self.compute_tokens(messages)
        self.tokens_output += self.compute_tokens([AIMessage(content=all_text)])

        # save in history input and output
        self.history.append(HumanMessage(content=last_request))
        self.history.append(AIMessage(content=all_text))

    @line_magic
    def clear_history(self, line):
        """
        Clear the conversation history.

        Args:
            line (str): Additional arguments (unused).
        """
        self.history = []
        logger.info("History cleared !")

    @line_magic
    def clear_stats(self, line):
        """
        Clear the genai stats.

        Args:
            line (str): Additional arguments (unused).
        """
        self.tokens_input = 0
        self.tokens_output = 0
        # to compute genai resp.time
        self.genai_requests = 0
        self.genai_total_time = 0
        logger.info("Stats cleared !")

    @line_magic
    def ask(self, line):
        """
        Send a query to the AI model and print the response.

        Args:
            line (str): The user's query.
        """
        messages = [
            SystemMessage(content=PROMPT_ASK),
            *self.history[-MAX_MSGS_IN_HISTORY:],
            HumanMessage(content=line),
        ]

        # send the messages to the model and print the response
        # we send separately line to save in history user request
        self.handle_input(messages, line)

    @line_magic
    def ask_code(self, line):
        """
        Request code generation from the AI model based on the current context and user input.

        Args:
            line (str): The user's request for code.
        """
        # get the variables in session
        context = get_context(self.shell.user_ns, line)
        # build input to the model
        messages = [
            SystemMessage(content=PROMPT_ASK_CODE),
            *self.history[-MAX_MSGS_IN_HISTORY:],
            HumanMessage(content=f"Context: {context}\n\n{line}"),
        ]
        # send the messages to the model and print the response
        self.handle_input(messages, line)

    @line_magic
    def ask_data(self, line):
        """
        Request data analysis from the AI model based on the current context and user input.

        Args:
            line (str): The user's request for data analysis.
        """
        context = get_context(self.shell.user_ns, line)

        # add the context
        messages = [
            SystemMessage(content=PROMPT_ASK_DATA),
            *self.history[-MAX_MSGS_IN_HISTORY:],
            HumanMessage(content=f"Context: {context}\n\n{line}"),
        ]
        # send the messages to the model and print the response
        self.handle_input(messages, line)

    @line_magic
    def show_variables(self, line):
        """
        Display the list of non-private variables in the current Jupyter Notebook session.

        Args:
            line (str): Additional arguments (unused).
        """
        user_ns = self.shell.user_ns

        variables_and_values = filter_variables(user_ns)

        print("User-defined variables in the current session:")
        for name, value in variables_and_values.items():
            print(f"* {name} (type: {type(value).__name__}): {value}")

    @line_magic
    def show_model_config(self, line):
        """
        Display the current OCI model configuration.

        Args:
            line (str): Additional arguments (unused).
        """
        print("Model configuration defined in config.py:")
        print("* Model: ", MODEL_ID)
        print("* Endpoint: ", SERVICE_ENDPOINT)
        print("* Temperature: ", TEMPERATURE)
        print("* Max_tokens: ", MAX_TOKENS)

    @line_magic
    def genai_stats(self, line):
        """
        Display the current OCI model performance.

        Args:
            line (str): Additional arguments (unused).
        """
        print("Performance metrics:")
        print("* Total requests: ", self.genai_requests)
        print("* Total input tokens: ", self.tokens_input)
        print("* Total output tokens: ", self.tokens_output)
        print(
            "* Avg time (sec.): ", round(self.genai_total_time / self.genai_requests, 1)
        )


def load_ipython_extension(ipython):
    """
    Load the OCIGenaiMagics extension into the IPython environment.

    Args:
        ipython (InteractiveShell): The current IPython shell instance.
    """
    print("OCIGenaiMagics extension loaded...")

    command_list = [
        "ask",
        "ask_code",
        "ask_data",
        "clear_history",
        "show_variables",
        "show_model_config",
        "genai_stats",
        "clear_stats",
    ]
    print("List of magic commands available:")
    for command in command_list:
        print(f"* {command}")

    ipython.register_magics(OCIGenaiMagics)
