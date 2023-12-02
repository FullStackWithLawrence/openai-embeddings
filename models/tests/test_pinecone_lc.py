# -*- coding: utf-8 -*-
# flake8: noqa: F401
"""
Test the pinecone OEM library and the Langchaine Pinecone wrapper class.
"""

import pinecone
import pytest  # pylint: disable=unused-import
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone

from models.const import Config, Credentials
from models.hybrid_search_retreiver import HybridSearchRetriever


class TestPinecone:
    """Test HybridSearchRetriever class."""

    def test_01_test_pinecone_connectivity(self):
        """Ensure that we have connectivity to Pinecone."""
        # pylint: disable=broad-except
        try:
            pinecone.init(api_key=Credentials.PINECONE_API_KEY, environment=Config.PINECONE_ENVIRONMENT)
        except Exception as e:
            assert False, f"pinecone.init() failed with exception: {e}"

    def test_02_test_pinecone_index(self):
        """Ensure that the Pinecone index exists and that we can connect to it."""

        # this is a no-op to ensure that the Pinecone index is initialized
        hsr = HybridSearchRetriever()
        index = hsr.pinecone.index
        assert isinstance(index, pinecone.Index)

        pinecone.init(api_key=Credentials.PINECONE_API_KEY, environment=Config.PINECONE_ENVIRONMENT)
        openai_embedding = OpenAIEmbeddings(
            api_key=Credentials.OPENAI_API_KEY, organization=Credentials.OPENAI_API_ORGANIZATION
        )

        # pylint: disable=broad-except
        try:
            Pinecone.from_existing_index(
                Config.PINECONE_INDEX_NAME,
                embedding=openai_embedding,
            )
        except Exception as e:
            assert False, f"Pinecone initialization of index {Config.PINECONE_INDEX_NAME,} failed with exception: {e}"
