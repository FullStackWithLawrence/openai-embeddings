# -*- coding: utf-8 -*-
# flake8: noqa: F401
"""
Test integrity of base class.
"""

import pinecone
import pytest  # pylint: disable=unused-import
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone

from ..const import Credentials


class TestPinecone:
    """Test SalesSupportModel class."""

    def test_01_test_pinecone_connectivity(self):
        """Ensure that we have connectivity to Pinecone."""
        # pylint: disable=broad-except
        try:
            pinecone.init(api_key=Credentials.PINECONE_API_KEY, environment=Credentials.PINECONE_ENVIRONMENT)
        except Exception as e:
            assert False, f"pinecone.init() failed with exception: {e}"

    def test_02_test_pinecone_index(self):
        """Ensure that the Pinecone index exists and that we can connect to it."""
        pinecone.init(api_key=Credentials.PINECONE_API_KEY, environment=Credentials.PINECONE_ENVIRONMENT)
        openai_embedding = OpenAIEmbeddings()

        # pylint: disable=broad-except
        try:
            Pinecone.from_existing_index(
                Credentials.PINECONE_INDEX_NAME,
                embedding=openai_embedding,
            )
        except Exception as e:
            assert (
                False
            ), f"Pinecone initialization of index {Credentials.PINECONE_INDEX_NAME,} failed with exception: {e}"
