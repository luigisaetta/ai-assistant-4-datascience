"""
Functions to access to OCI Genai models
"""

from langchain_community.chat_models import ChatOCIGenAI

from config import (
    MODEL_ID,
    AUTH,
    SERVICE_ENDPOINT,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    COMPARTMENT_ID,
)


def get_llm():
    """
    Initialize and return an instance of ChatOCIGenAI with the specified configuration.

    Returns:
        ChatOCIGenAI: An instance of the OCI GenAI language model.
    """
    llm = ChatOCIGenAI(
        auth_type=AUTH,
        model_id=MODEL_ID,
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        is_stream=True,
        model_kwargs={
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "top_p": TOP_P,
        },
    )
    return llm
