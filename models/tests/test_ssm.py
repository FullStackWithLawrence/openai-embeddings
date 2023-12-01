# -*- coding: utf-8 -*-
# flake8: noqa: F401
"""
Test integrity of base class.
"""
import pytest  # pylint: disable=unused-import
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone

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
