# -*- coding: utf-8 -*-
"""Setup for openai_embeddings package."""
from setuptools import find_packages, setup

from setup_utils import get_semantic_version  # pylint: disable=import-error


setup(
    name="openai_embeddings",
    version=get_semantic_version(),
    description="""A Hybrid Search and Augmented Generation prompting solution using
    Python [OpenAI](https://openai.com/) embeddings sourced from
    [Pinecone](https://docs.pinecone.io/docs/python-client) vector database indexes and
    managed by [LangChain](https://www.langchain.com/).""",
    author="Lawrence McDaniel",
    author_email="lpm0073@gmail.com",
    packages=find_packages(),
    package_data={
        "openai_embeddings": ["*.md"],
    },
    install_requires=[
        "langchain>=0.2",
        "langchainhub>=0.1.14",
        "langchain-experimental",
        "openai>=1.40",
        "pinecone-client>=5",
        "pinecone-text>=0.9",
        "pydantic>=2.10",
        "pypdf>=5",
        "python-dotenv>=1.0.0",
        "tiktoken>=0.8",
    ],
)
