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
        "langchain>=0.0.341",
        "langchainhub>=0.1.14",
        "langchain-experimental>=0.0.43",
        "openai>=1.3.5",
        "pinecone-client>=2.2.4",
        "pinecone-text>=0.7.0",
        "pydantic>=2.0.0",
        "pypdf>=3.17.1",
        "python-dotenv>=1.0.0",
        "tiktoken>=0.5.1",
    ],
)
