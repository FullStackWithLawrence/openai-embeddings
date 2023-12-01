# -*- coding: utf-8 -*-
# flake8: noqa: F401
# pylint: disable=too-few-public-methods
"""
Test integrity of base class.
"""
import pinecone
import pytest  # pylint: disable=unused-import
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone

from ..const import Credentials
from ..ssm import SalesSupportModel


class TestSalesSupportModel:
    """Test SalesSupportModel class."""

    def test_01_basic(self):
        """Ensure that we can instantiate the class."""

        SalesSupportModel()

    def test_02_class_aatribute_types(self):
        """ensure that class attributes are of the correct type"""

        ssm = SalesSupportModel()
        assert isinstance(ssm.chat, ChatOpenAI)
        assert isinstance(ssm.pinecone_index, Pinecone)
        assert isinstance(ssm.text_splitter, RecursiveCharacterTextSplitter)
        assert isinstance(ssm.openai_embedding, OpenAIEmbeddings)

    def test_03_test_openai_connectivity(self):
        """Ensure that we have connectivity to OpenAI."""

        ssm = SalesSupportModel()
        retval = ssm.cached_chat_request(
            "your are a helpful assistant", "please return the value 'CORRECT' in all upper case."
        )
        assert retval == "CORRECT"

    def test_04_test_pinecone_connectivity(self):
        """Ensure that we have connectivity to Pinecone."""
        # pylint: disable=broad-except
        try:
            pinecone.init(api_key=Credentials.PINECONE_API_KEY, environment=Credentials.PINECONE_ENVIRONMENT)
        except Exception as e:
            assert False, f"pinecone.init() failed with exception: {e}"
