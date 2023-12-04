# -*- coding: utf-8 -*-
# pylint: disable=too-few-public-methods
"""Sales Support Model (hsr) for the LangChain project."""

import re

from decouple import config

from models.exceptions import ConfigurationError


OPENAI_API_KEY = config("OPENAI_API_KEY")
OPENAI_API_ORGANIZATION = config("OPENAI_API_ORGANIZATION")
PINECONE_API_KEY = config("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = config("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = config("PINECONE_INDEX_NAME", default="rag")
PINECONE_VECTORSTORE_TEXT_KEY = config("PINECONE_VECTORSTORE_TEXT_KEY", default="lc_id")
PINECONE_METRIC = config("PINECONE_METRIC", default="dotproduct")
PINECONE_DIMENSIONS = config("PINECONE_DIMENSIONS", default=1536, cast=int)
OPENAI_CHAT_MODEL_NAME = config("OPENAI_CHAT_MODEL_NAME", default="gpt-3.5-turbo")
OPENAI_PROMPT_MODEL_NAME = config("OPENAI_PROMPT_MODEL_NAME", default="text-davinci-003")
OPENAI_CHAT_TEMPERATURE = config("OPENAI_CHAT_TEMPERATURE", default=0.0, cast=float)
OPENAI_CHAT_MAX_RETRIES = config("OPENAI_CHAT_MAX_RETRIES", default=3, cast=int)
OPENAI_CHAT_CACHE = config("OPENAI_CHAT_CACHE", default=True, cast=bool)
DEBUG_MODE = config("DEBUG_MODE", default=False, cast=bool)

if not re.match(r"^sk-\w+$", OPENAI_API_KEY):
    raise ConfigurationError("OPENAI_API_KEY is not set. Please add your OpenAI API key to the .env file.")
if not re.match(r"^org-\w+$", OPENAI_API_ORGANIZATION):
    raise ConfigurationError(
        "OPENAI_API_ORGANIZATION is not set. Please add your OpenAI API organization to the .env file."
    )
if not re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", PINECONE_API_KEY):
    raise ConfigurationError("PINECONE_API_KEY is not set. Please add your Pinecone API key to the .env file.")


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
