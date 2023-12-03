# -*- coding: utf-8 -*-
# pylint: disable=too-few-public-methods
"""Sales Support Model (hsr) for the LangChain project."""

import os
import re

from dotenv import find_dotenv, load_dotenv


class ConfigurationError(Exception):
    """Exception raised for errors in the configuration."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


# pylint: disable=duplicate-code
dotenv_path = find_dotenv()
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, verbose=True)
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    OPENAI_API_ORGANIZATION = os.environ["OPENAI_API_ORGANIZATION"]
    PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
    PINECONE_ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]
    PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "hsr")
    PINECONE_VECTORSTORE_TEXT_KEY = os.environ.get("PINECONE_VECTORSTORE_TEXT_KEY", "lc_id")
    PINECONE_METRIC = os.environ.get("PINECONE_METRIC", "dotproduct")
    PINECONE_DIMENSIONS = int(os.environ.get("PINECONE_DIMENSIONS", 1536))
    OPENAI_CHAT_MODEL_NAME = os.environ.get("OPENAI_CHAT_MODEL_NAME", "gpt-3.5-turbo")
    OPENAI_PROMPT_MODEL_NAME = os.environ.get("OPENAI_PROMPT_MODEL_NAME", "text-davinci-003")
    OPENAI_CHAT_TEMPERATURE = float(os.environ.get("OPENAI_CHAT_TEMPERATURE", 0.0))
    OPENAI_CHAT_MAX_RETRIES = int(os.environ.get("OPENAI_CHAT_MAX_RETRIES", 3))
    OPENAI_CHAT_CACHE = bool(os.environ.get("OPENAI_CHAT_CACHE", True))
    DEBUG_MODE = os.environ.get("DEBUG_MODE", "False") == "True"

    if not re.match(r"^sk-\w+$", OPENAI_API_KEY):
        raise ConfigurationError("OPENAI_API_KEY is not set. Please add your OpenAI API key to the .env file.")
    if not re.match(r"^org-\w+$", OPENAI_API_ORGANIZATION):
        raise ConfigurationError(
            "OPENAI_API_ORGANIZATION is not set. Please add your OpenAI API organization to the .env file."
        )
    if not re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", PINECONE_API_KEY):
        raise ConfigurationError("PINECONE_API_KEY is not set. Please add your Pinecone API key to the .env file.")

else:
    raise FileNotFoundError("No .env file found in root directory of repository")


class ReadOnly(type):
    """Metaclass to make all class attributes read-only."""

    def __setattr__(cls, name, value):
        if name in cls.__dict__:
            raise TypeError(f"Cannot change a read-only attribute {name}")
        super().__setattr__(name, value)


class Config(metaclass=ReadOnly):
    """Configuration parameters."""

    DEBUG_MODE: bool = DEBUG_MODE
    OPENAI_CHAT_MODEL_NAME: str = OPENAI_CHAT_MODEL_NAME
    OPENAI_PROMPT_MODEL_NAME: str = OPENAI_PROMPT_MODEL_NAME
    OPENAI_CHAT_TEMPERATURE: float = OPENAI_CHAT_TEMPERATURE
    OPENAI_CHAT_MAX_RETRIES: int = OPENAI_CHAT_MAX_RETRIES
    OPENAI_CHAT_CACHE: bool = OPENAI_CHAT_CACHE
    PINECONE_ENVIRONMENT = PINECONE_ENVIRONMENT
    PINECONE_INDEX_NAME = PINECONE_INDEX_NAME
    PINECONE_VECTORSTORE_TEXT_KEY: str = PINECONE_VECTORSTORE_TEXT_KEY
    PINECONE_METRIC: str = PINECONE_METRIC
    PINECONE_DIMENSIONS: int = PINECONE_DIMENSIONS


class Credentials(metaclass=ReadOnly):
    """Credentials."""

    OPENAI_API_KEY = OPENAI_API_KEY
    OPENAI_API_ORGANIZATION = OPENAI_API_ORGANIZATION
    PINECONE_API_KEY = PINECONE_API_KEY
