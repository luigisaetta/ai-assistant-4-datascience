"""
CSV analyzer

- analyze a CSV and answer a given question
- load the csv in a pandas dataframe
- generate the code
- execute the code

"""

import sys
import io
import pandas as pd

from langchain_core.messages import HumanMessage, SystemMessage
from code_parser_utils import remove_triple_backtics
from context import get_variable_info
from oci_models import get_llm
from prompts import PROMPT_ASK_CODE

DEBUG = True

def read_csv(file):
    """
    read the csv file and return a pandas dataframe
    """
    return pd.read_csv(file)


def generate_code(df, question):
    """
    generate the code, to be executed on the df, to answer to the question
    """
    llm = get_llm()

    df_info = get_variable_info("df", df)

    # print(df_info)

    CONTEXT_AND_REQUEST = f"""
    Context: {df_info}\n
    Question: {question}
    """
    messages = [
        SystemMessage(content=PROMPT_ASK_CODE),
        HumanMessage(content=CONTEXT_AND_REQUEST),
    ]

    _response = llm.invoke(messages)

    _code = remove_triple_backtics(_response.content)

    if DEBUG:
        print("Generated code: ")
        print(_code)
        print("")

    return _code


def exec_code(_df, code):
    """
    execute the code
    """
    # since the exec work on df, we need to have it here
    df = _df.copy()

    output_capture = io.StringIO()

    # Redirect to StringIO
    sys.stdout = output_capture

    # exec the code
    exec(code)

    # Ripristiniamo sys.stdout al valore originale
    sys.stdout = sys.__stdout__

    # Otteniamo l'output catturato come stringa
    captured_output = output_capture.getvalue()

    return captured_output


def generate_answer(question, code_result):
    """
    generate the answer to the question
    """
    llm = get_llm()

    SYSTEM_PROMPT = f"""
    Generate a clear and concise summary that includes both the provided question and its corresponding answer.

    Use only the information provided in the context without adding external details.
    Never indicate that information is missing or insufficient. Assume all necessary information is present.
    If explicit details are not provided, generate the most relevant and reasonable response using what is available.
    Do not state that the context "does not mention" or "does not contain" something. Instead, 
    focus on presenting a response using the given content.
    Structure the summary in a clear and professional manner, using an informative header.
    Ensure the language is formal, precise, and easy to understand.

    """

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context: {code_result}\nQuestion: {question}\n"),
    ]

    return llm.invoke(messages).content


def process_request(f_name, question):
    """
    Main function to process the request
    """
    df = read_csv(f_name)

    code = generate_code(df, question)

    captured_output = exec_code(df, code)

    return generate_answer(question, captured_output)


#
# Main
#
F_NAME = "travel_hospitality.csv"

QUESTION = """Show me the list of restaurants in London with rating > 3.5,
display the output in a nicely formatted table"""
# QUESTION = "give me the Name of restaurant in New Your that Fabricio has recommended"

print(process_request(F_NAME, QUESTION))
print("")
