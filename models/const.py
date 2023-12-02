# -*- coding: utf-8 -*-
# pylint: disable=too-few-public-methods
"""Sales Support Model (SSM) for the LangChain project."""

import os

from dotenv import find_dotenv, load_dotenv


# pylint: disable=duplicate-code
dotenv_path = find_dotenv()
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, verbose=True)
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    OPENAI_API_ORGANIZATION = os.environ["OPENAI_API_ORGANIZATION"]
    PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
    PINECONE_ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]
    PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "hsr")
    OPENAI_CHAT_MODEL_NAME = os.environ.get("OPENAI_CHAT_MODEL_NAME", "gpt-3.5-turbo")
    OPENAI_PROMPT_MODEL_NAME = os.environ.get("OPENAI_PROMPT_MODEL_NAME", "text-davinci-003")
else:
    raise FileNotFoundError("No .env file found in root directory of repository")


class Config:
    """Configuration parameters."""

    OPENAI_CHAT_MODEL_NAME = OPENAI_CHAT_MODEL_NAME
    OPENAI_PROMPT_MODEL_NAME = OPENAI_PROMPT_MODEL_NAME


class Credentials:
    """Credentials."""

    OPENAI_API_KEY = OPENAI_API_KEY
    OPENAI_API_ORGANIZATION = OPENAI_API_ORGANIZATION
    PINECONE_API_KEY = PINECONE_API_KEY
    PINECONE_ENVIRONMENT = PINECONE_ENVIRONMENT
    PINECONE_INDEX_NAME = PINECONE_INDEX_NAME
