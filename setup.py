# -*- coding: utf-8 -*-
"""
Future use: setup for openai_embeddings package. I use this for instructional purposes,
for demonstrating best practices on how to create a Python package.

This package is not actually published to PyPi.
"""
import io
import os
from typing import List

from setuptools import find_packages, setup

from setup_utils import get_semantic_version  # pylint: disable=import-error


HERE = os.path.abspath(os.path.dirname(__file__))


def is_requirement(line: str) -> bool:
    """
    True if line is a valid requirement line from a
    Python requirements file.
    """
    return not (line.strip() == "" or line.startswith("#"))


def load_requirements(filename: str) -> List[str]:
    """
    Returns Python package requirements as a list of semantically
    versioned pip packages.

    Args:
        filename: The name of the requirements file to load. example: "base.txt"

    Returns:
        A list of package requirements.
        ['pytest==8.3.4', 'pytest_mock==3.14.0', 'black==25.1.0', ... more packages ]
    """
    with io.open(os.path.join(HERE, "requirements", filename), "rt", encoding="utf-8") as f:
        return [line.strip() for line in f if is_requirement(line) and not line.startswith("-r")]


setup(
    name="openai_embeddings",
    version=get_semantic_version(),
    description="""A Hybrid Search and Augmented Generation prompting solution using
    Python [OpenAI](https://openai.com/) embeddings sourced from
    [Pinecone](https://docs.pinecone.io/docs/python-client) vector database indexes and
    managed by [LangChain](https://www.langchain.com/).""",
    author="Lawrence McDaniel",
    author_email="lpm0073@gmail.com",
    url="https://lawrencemcdaniel.com/",
    packages=find_packages(),
    package_data={
        "openai_embeddings": ["*.md"],
    },
    install_requires=load_requirements("base.txt"),
    extras_require={
        "dev": load_requirements("local.txt"),
    },
)
