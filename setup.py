# -*- coding: utf-8 -*-
"""Setup for netec_models package."""
from setuptools import find_packages, setup

from setup_utils import get_semantic_version  # pylint: disable=import-error


setup(
    name="netec_models",
    version=get_semantic_version(),
    description="Netec Sales Support Model (SSM)",
    author="Lawrence McDaniel",
    author_email="lpm0073@gmail.com",
    packages=find_packages(),
    package_data={
        "netec_models": ["*.md"],
    },
    install_requires=["pydantic>=2.0.0", "python-dotenv>=1.0.0", "langchain>=0.0.341", "pinecone-client>=2.2.4"],
)
