"""
This module implements magic commands for integrating an OCI GenAI model
within a Jupyter Notebook.

inspired by: https://github.com/vinayak-mehta/ipychat
"""

from IPython.core.magic import Magics, line_magic, magics_class
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from oci_models import get_llm
from context import filter_variables, get_context
from prompts import PROMPT_ASK, PROMPT_ASK_CODE, PROMPT_ASK_DATA

from config import (
    MODEL_ID,
    SERVICE_ENDPOINT,
    MAX_TOKENS,
    TEMPERATURE,
)


@magics_class
class OCIGenaiMagics(Magics):
    """
    A class to implement custom magic commands for interacting with an OCI GenAI model.
    """

    def __init__(self, shell):
        """
        Initialize a new instance of MyClass.
        """
        super().__init__(shell)
        # the list of messages
        self.history = []

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

    @line_magic
    def clear_history(self, line):
        """
        Clear the conversation history.

        Args:
            line (str): Additional arguments (unused).
        """
        self.history = []
        print("History cleared !")

    @line_magic
    def ask(self, line):
        """
        Send a query to the AI model and print the response.

        Args:
            line (str): The user's query.
        """
        llm = get_llm()

        messages = [
            SystemMessage(content=PROMPT_ASK),
            *self.history,
            HumanMessage(content=line),
        ]

        ai_response = llm.stream(messages)

        all_text = self.print_stream(ai_response)

        # save in history input and output
        self.history.append(HumanMessage(content=line))
        self.history.append(AIMessage(content=all_text))

    @line_magic
    def ask_code(self, line):
        """
        Request code generation from the AI model based on the current context and user input.

        Args:
            line (str): The user's request for code.
        """
        llm = get_llm()

        # get the variables in session
        context = get_context(self.shell.user_ns, line)
        # build input to the model
        messages = [
            SystemMessage(content=PROMPT_ASK_CODE),
            *self.history,
            HumanMessage(content=f"Context: {context}\n\n{line}"),
        ]
        # invoke the model
        ai_response = llm.stream(messages)
        # print in streaming mode
        all_text = self.print_stream(ai_response)

        # save in history input and output
        self.history.append(HumanMessage(content=line))
        self.history.append(AIMessage(content=all_text))

    @line_magic
    def ask_data(self, line):
        """
        Request data analysis from the AI model based on the current context and user input.

        Args:
            line (str): The user's request for data analysis.
        """
        llm = get_llm()

        context = get_context(self.shell.user_ns, line)

        # add the context
        messages = [
            SystemMessage(content=PROMPT_ASK_DATA),
            HumanMessage(content=f"Context: {context}\n\n{line}"),
        ]

        ai_response = llm.stream(messages)

        self.print_stream(ai_response)

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
            print(f"* {name}: {value}")

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
    ]
    print("List of magic commands available:")
    for command in command_list:
        print(f"* {command}")

    ipython.register_magics(OCIGenaiMagics)
